//	120210400 조 용 현
//  AoS_SO_kernel.cl
//
//  Written for CSEG437_CSE5437
//  Department of Computer Science and Engineering
//  Copyright © 2021 Sogang University. All rights reserved.

__kernel void AoS_SO_kernel(
	const __global uchar* input_data, __global uchar* midput_data, __global uchar* output_data,
	int n_columns, int n_rows)                          // image width & height
{
    // (1) Grayscale 진행.
	int column = get_global_id(0);
	int row = get_global_id(1);
    int i = 4 * (row * n_columns + column);

    uchar intensity = (uchar)(0.299f * input_data[i]    // R
            + 0.587f * input_data[i+1]                  // G
            + 0.114f * input_data[i+2]);                // B

    midput_data[i] = intensity;
//      Grayscale를 거친 결과는, 각 R,G,B에 들어 있는 값들은 모두 동일하므로, R 필드에 대한 연산만 수행해도 문제가 없다.
//      midput_data[i+1] = intensity;
//      midput_data[i+2] = intensity;
    midput_data[i+3] = input_data[i+3];

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
                elem = (int)(midput_data[i + 4 * ((k) * n_columns + (l))]);
                Gx += elem * Sx[1 + k][1 + l];
                Gy += elem * Sy[1 + k][1 + l];
            }
        }
    }
    pixel = (int) sqrt((double)Gx * Gx + (double)Gy * Gy);

    // R 필드에 대한 값을 나머지 G, B 필드에도 Broadcasting 해준다.
    output_data[i] = (uchar)pixel;
    output_data[i+1] = (uchar)pixel;
    output_data[i+2] = (uchar)pixel;
    output_data[i+3] = midput_data[i+3];
}