//	120210400 조 용 현
//  SoA_SO_kernel.cl
//
//  Written for CSEG437_CSE5437
//  Department of Computer Science and Engineering
//  Copyright © 2021 Sogang University. All rights reserved.

__kernel void SoA_SO_kernel(
	const __global uchar* input_data_R, __global uchar* midput_data_R, __global uchar* output_data_R,
    const __global uchar* input_data_G, __global uchar* midput_data_G, __global uchar* output_data_G,
    const __global uchar* input_data_B, __global uchar* midput_data_B, __global uchar* output_data_B,
    const __global uchar* input_data_A, __global uchar* midput_data_A, __global uchar* output_data_A,
	int n_columns, int n_rows)                          // image width & height
{
    // (1) Grayscale 진행.
    int column = get_global_id(0);
    int row = get_global_id(1);
    int i = (row * n_columns + column);

    uchar intensity = (uchar)(0.299f * input_data_R[i]  // R
            + 0.587f * input_data_G[i]                  // G
            + 0.114f * input_data_B[i]);                // B

    midput_data_R[i] = intensity;
//      Grayscale를 거친 결과는, 각 R,G,B에 들어 있는 값들은 모두 동일하므로, R 필드에 대한 연산만 수행해도 문제가 없다.
//      midput_data_G[i] = intensity;
//      midput_data_B[i] = intensity;
    midput_data_A[i] = input_data_A[i];

    // (2) 매 픽셀마다 sobel operation 필터 적용.
    int Sx[3][3] = { {1,0,-1}, {2,0,-2}, {1,0,-1} };
    int Sy[3][3] = { {1,2,1}, {0,0,0}, {-1,-2,-1} };

    int k, l;
    int elem;
    int Gx, Gy;
    double pixel;

    elem = 0;
    pixel = 0;
    Gx = 0; Gy = 0;

    // 커널 2차원 indexing.
    for (k = -1; k <= 1; k++) {
        for (l = -1; l <= 1; l++) {
            // Out of range 즉, 패딩인 부분은 무시하는 조건문.
            if ((row + k) >= 0 && (column + l) >= 0 && (row + k) < n_rows && (column + l) < n_columns) {
                // Grayscale를 거친 결과는, 각 R,G,B에 들어 있는 값들은 모두 동일하므로, R 필드에 대한 연산만 수행해도 문제가 없다.
                elem = (int)(midput_data_R[i + (k) * n_columns + (l)]);
                Gx += elem * Sx[1 + k][1 + l];
                Gy += elem * Sy[1 + k][1 + l];
            }
        }
    }
    pixel = (int) sqrt((double)Gx * Gx + (double)Gy * Gy);

    // R 필드에 대한 값을 나머지 G, B 필드에도 Broadcasting 해준다.
    output_data_R[i] = (uchar)pixel;
    output_data_G[i] = (uchar)pixel;
    output_data_B[i] = (uchar)pixel;
    output_data_A[i] = midput_data_A[i];
}