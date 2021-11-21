//  120210400 조 용 현

#include "Context_SoA_AoS.h"

// Sobel Operator Kernel.
int Sx[3][3] = { {1,0,-1}, {2,0,-2}, {1,0,-1} };
int Sy[3][3] = { {1,2,1}, {0,0,0}, {-1,-2,-1} };

void convert_to_greyscale_image_SoA_CPU(void) {
    for (unsigned int i = 0; i < context.image_width * context.image_height; i++) {

        BYTE intensity = BYTE(0.299f * context.SoA_image_input.R_plane[i]  // R
            + 0.587f * context.SoA_image_input.G_plane[i]  // G
            + 0.114f * context.SoA_image_input.B_plane[i]);  // B
        context.SoA_image_output.R_plane[i] = intensity;
        context.SoA_image_output.G_plane[i] = intensity;
        context.SoA_image_output.B_plane[i] = intensity;
        context.SoA_image_output.A_plane[i] = context.SoA_image_input.A_plane[i];
    }
}

void convert_to_greyscale_image_AoS_CPU(void) {
    Pixel_Channels* tmp_ptr_input = context.AoS_image_input;
    Pixel_Channels* tmp_ptr_output = context.AoS_image_output;
    for (unsigned int i = 0; i < context.image_width * context.image_height; i++) {
        BYTE intensity = BYTE(0.299f * tmp_ptr_input->R // R
            + 0.587f * tmp_ptr_input->G  // G
            + 0.114f * tmp_ptr_input->B);  // B
        tmp_ptr_output->R = intensity;
        tmp_ptr_output->G = intensity;
        tmp_ptr_output->B = intensity;
        tmp_ptr_output->A = tmp_ptr_input->A;

        tmp_ptr_input++; tmp_ptr_output++;
    }
}

// [SoA_SO_CPU]
void apply_sobel_operator_to_image_SoA_CPU(void) {
    // (1) Grayscale 진행.
    for (unsigned int i = 0; i < context.image_width * context.image_height; i++) {

        BYTE intensity = BYTE(0.299f * context.SoA_image_input.R_plane[i]  // R
            + 0.587f * context.SoA_image_input.G_plane[i]  // G
            + 0.114f * context.SoA_image_input.B_plane[i]);  // B
        context.SoA_image_midput.R_plane[i] = intensity;
        context.SoA_image_midput.G_plane[i] = intensity;
        context.SoA_image_midput.B_plane[i] = intensity;
        context.SoA_image_midput.A_plane[i] = context.SoA_image_input.A_plane[i];
    }

    // (2) 매 픽셀마다 sobel operation 필터 적용.
    int r = 0;
    int c = 0;
    int k, l;

    int elem;
    int Gx, Gy;
    double pixel;

    for (r = 0; r < context.image_height; r++)
    {
        for (c = 0; c < context.image_width; c++)
        {
            elem = 0;
            pixel = 0;
            Gx = 0; Gy = 0;
            // 커널 2차원 indexing.
            for (k = -1; k <= 1; k++) {
                for (l = -1; l <= 1; l++) {
                    // Out of range 즉, 패딩인 부분은 무시하는 조건문.
                    if (((r + k) >= 0 && (c + l) >= 0) && ((r + k) < context.image_height && (c + l) < context.image_width)) {
                        // Grayscale를 거친 결과는, 각 R,G,B에 들어 있는 값들은 모두 동일하므로, R 필드에 대한 연산만 수행해도 문제가 없다.
                        elem = (int)(context.SoA_image_midput.R_plane[(r + k) * context.image_width + (c + l)]);
                        Gx += elem * Sx[1 + k][1 + l];  // printf("Gx: %d \n", Gx);
                        Gy += elem * Sy[1 + k][1 + l];  // printf("Gy: %d \n", Gy);
                    }
                }
            }
            pixel = sqrt((double)Gx * Gx + (double)Gy * Gy);

            // R 필드에 대한 값을 나머지 G, B 필드에도 Broadcasting 해준다.
            context.SoA_image_output.R_plane[r * context.image_width + c] = (BYTE)pixel;
            context.SoA_image_output.G_plane[r * context.image_width + c] = (BYTE)pixel;
            context.SoA_image_output.B_plane[r * context.image_width + c] = (BYTE)pixel;
            context.SoA_image_output.A_plane[r * context.image_width + c] = context.SoA_image_midput.A_plane[r * context.image_width + c];
        }
    }
}

// [AoS_SO_CPU]
void apply_sobel_operator_to_image_AoS_CPU(void) {
    // (1) Grayscale 진행.
    Pixel_Channels* tmp_ptr_input = context.AoS_image_input;
    Pixel_Channels* tmp_ptr_midput = context.AoS_image_midput;
    Pixel_Channels* tmp_ptr_output = context.AoS_image_output;

    for (unsigned int i = 0; i < context.image_width * context.image_height; i++) {
        BYTE intensity = BYTE(0.299f * tmp_ptr_input->R // R
            + 0.587f * tmp_ptr_input->G  // G
            + 0.114f * tmp_ptr_input->B);  // B
        tmp_ptr_midput->R = intensity;
        tmp_ptr_midput->G = intensity;
        tmp_ptr_midput->B = intensity;
        tmp_ptr_midput->A = tmp_ptr_input->A;

        tmp_ptr_input++; tmp_ptr_midput++;
    }

    // (2) 매 픽셀마다 sobel operation 필터 적용.
    int r = 0;
    int c = 0;
    int k, l;

    int elem;
    int Gx, Gy;
    double pixel;
     
    // 중간 결과를 저장하는 메모리영역의 pointer를 맨 앞으로 reposition시킨다.
    tmp_ptr_midput = context.AoS_image_midput;

    for (r = 0; r < context.image_height; r++)
    {
        for (c = 0; c < context.image_width; c++)
        {
            elem = 0;
            pixel = 0;
            Gx = 0; Gy = 0;
            // 커널 2차원 indexing.
            for (k = -1; k <= 1; k++) {
                for (l = -1; l <= 1; l++) {
                    // Out of range 즉, 패딩인 부분은 무시하는 조건문.
                    if (((r + k) >= 0 && (c + l) >= 0) && ((r + k) < context.image_height && (c + l) < context.image_width)) {
                        // Grayscale를 거친 결과는, 각 R,G,B에 들어 있는 값들은 모두 동일하므로, R 필드에 대한 연산만 수행해도 문제가 없다.
                        elem = (int)(tmp_ptr_midput + (r + k) * context.image_width + (c + l))->R;
                        Gx += elem * Sx[1 + k][1 + l];  // printf("Gx: %d \n", Gx);
                        Gy += elem * Sy[1 + k][1 + l];  // printf("Gy: %d \n", Gy);
                    }
                }
            }
            pixel = sqrt((double)Gx * Gx + (double)Gy * Gy);

            (tmp_ptr_output + r * context.image_width + c)->R = (BYTE)pixel;
            (tmp_ptr_output + r * context.image_width + c)->G = (BYTE)pixel;
            (tmp_ptr_output + r * context.image_width + c)->B = (BYTE)pixel;
            (tmp_ptr_output + r * context.image_width + c)->A = tmp_ptr_midput->A;
        }
    }
}