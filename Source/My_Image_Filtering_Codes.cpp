//  120210400 �� �� ��

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
    // (1) Grayscale ����.
    for (unsigned int i = 0; i < context.image_width * context.image_height; i++) {

        BYTE intensity = BYTE(0.299f * context.SoA_image_input.R_plane[i]  // R
            + 0.587f * context.SoA_image_input.G_plane[i]  // G
            + 0.114f * context.SoA_image_input.B_plane[i]);  // B
        context.SoA_image_midput.R_plane[i] = intensity;
        context.SoA_image_midput.G_plane[i] = intensity;
        context.SoA_image_midput.B_plane[i] = intensity;
        context.SoA_image_midput.A_plane[i] = context.SoA_image_input.A_plane[i];
    }

    // (2) �� �ȼ����� sobel operation ���� ����.
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
            // Ŀ�� 2���� indexing.
            for (k = -1; k <= 1; k++) {
                for (l = -1; l <= 1; l++) {
                    // Out of range ��, �е��� �κ��� �����ϴ� ���ǹ�.
                    if (((r + k) >= 0 && (c + l) >= 0) && ((r + k) < context.image_height && (c + l) < context.image_width)) {
                        // Grayscale�� ��ģ �����, �� R,G,B�� ��� �ִ� ������ ��� �����ϹǷ�, R �ʵ忡 ���� ���길 �����ص� ������ ����.
                        elem = (int)(context.SoA_image_midput.R_plane[(r + k) * context.image_width + (c + l)]);
                        Gx += elem * Sx[1 + k][1 + l];  // printf("Gx: %d \n", Gx);
                        Gy += elem * Sy[1 + k][1 + l];  // printf("Gy: %d \n", Gy);
                    }
                }
            }
            pixel = sqrt((double)Gx * Gx + (double)Gy * Gy);

            // R �ʵ忡 ���� ���� ������ G, B �ʵ忡�� Broadcasting ���ش�.
            context.SoA_image_output.R_plane[r * context.image_width + c] = (BYTE)pixel;
            context.SoA_image_output.G_plane[r * context.image_width + c] = (BYTE)pixel;
            context.SoA_image_output.B_plane[r * context.image_width + c] = (BYTE)pixel;
            context.SoA_image_output.A_plane[r * context.image_width + c] = context.SoA_image_midput.A_plane[r * context.image_width + c];
        }
    }
}

// [AoS_SO_CPU]
void apply_sobel_operator_to_image_AoS_CPU(void) {
    // (1) Grayscale ����.
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

    // (2) �� �ȼ����� sobel operation ���� ����.
    int r = 0;
    int c = 0;
    int k, l;

    int elem;
    int Gx, Gy;
    double pixel;
     
    // �߰� ����� �����ϴ� �޸𸮿����� pointer�� �� ������ reposition��Ų��.
    tmp_ptr_midput = context.AoS_image_midput;

    for (r = 0; r < context.image_height; r++)
    {
        for (c = 0; c < context.image_width; c++)
        {
            elem = 0;
            pixel = 0;
            Gx = 0; Gy = 0;
            // Ŀ�� 2���� indexing.
            for (k = -1; k <= 1; k++) {
                for (l = -1; l <= 1; l++) {
                    // Out of range ��, �е��� �κ��� �����ϴ� ���ǹ�.
                    if (((r + k) >= 0 && (c + l) >= 0) && ((r + k) < context.image_height && (c + l) < context.image_width)) {
                        // Grayscale�� ��ģ �����, �� R,G,B�� ��� �ִ� ������ ��� �����ϹǷ�, R �ʵ忡 ���� ���길 �����ص� ������ ����.
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