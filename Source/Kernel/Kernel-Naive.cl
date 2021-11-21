//	120210400 조 용 현
//  Kernel-Naive.cl
//
//  Written for CSEG437_CSE5437
//  Department of Computer Science and Engineering
//  Copyright © 2021 Sogang University. All rights reserved.

__kernel void Kernel_Naive(
	const __global uchar* input_data, __global uchar* midput_data, __global uchar* output_data,
	int n_columns, int n_rows, __constant int* filter_x_weights, __constant int* filter_y_weights,
	int filter_x_width, int filter_y_width)
{    
	int column = get_global_id(0);
	int row = get_global_id(1);

	const int HFS = (int)(filter_x_width / 2); // Half Filter Size

	int Sx = 0;
	int Sy = 0;

	float gray[3] = {0.299f, 0.587f, 0.114f};
	int offset = 4 * (row * n_columns + column);

	int filter_x_index = 0;
	int filter_y_index = 0;
	for (int r = -HFS; r <= HFS; r++) {
		for (int c = -HFS; c <= HFS; c++) {
			if (row + r < 0 || row + r >= n_rows || column + c < 0 || column + c >= n_columns) continue;
			uchar intensity = (uchar)(input_data[offset + 4 * (r * n_columns + c)] * gray[0])
								+ (uchar)(input_data[offset + 1 + 4 * (r * n_columns + c)] * gray[1])
								+ (uchar)(input_data[offset + 2 + 4 * (r * n_columns + c)] * gray[2]);

			Sx += intensity * filter_x_weights[filter_x_index++];
            Sy += intensity * filter_y_weights[filter_y_index++];
		}
	}
	uchar rgb = (uchar) sqrt( (double)(Sx * Sx) + (double)(Sy * Sy) ) / 8;

	output_data[offset] = rgb;
	output_data[offset + 1] = rgb;
	output_data[offset + 2] = rgb;
	output_data[offset + 3] = input_data[offset+3];
}