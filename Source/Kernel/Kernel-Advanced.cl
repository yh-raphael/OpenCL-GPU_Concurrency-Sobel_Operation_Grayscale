//	120210400 조 용 현
//  Kernel-Advanced.cl						// Vector4 type variable + Loop unrolling + midput buffer.
//
//  Written for CSEG437_CSE5437
//  Department of Computer Science and Engineering
//  Copyright © 2021 Sogang University. All rights reserved.

__kernel void Kernel_Advanced(
	const __global uchar4* input_data, __global uchar4* midput_data, __global uchar4* output_data,
	int n_columns, int n_rows, __constant int* filter_x_weights, __constant int* filter_y_weights,
	int filter_x_width, int filter_y_width)
{    
	int column = get_global_id(0);
	int row = get_global_id(1);

	const int HFS = (int)(filter_x_width / 2); // Half Filter Size

	int Sx = 0;
	int Sy = 0;

	float gray[3] = {0.299f, 0.587f, 0.114f};
	int offset = row * n_columns + column;

	uchar intensity = (uchar)(input_data[offset].x * gray[0]) + (uchar)(input_data[offset].y * gray[1])
						+ (uchar)(input_data[offset].z * gray[2]);

	midput_data[offset].x = intensity;
	midput_data[offset].y = intensity;
	midput_data[offset].z = intensity;
	midput_data[offset].w = input_data[offset].w;

	int filter_x_index = 0;
	int filter_y_index = 0;
	for (int r = -HFS; r <= HFS; r++) {
			//for (int c = -HFS; c <= HFS; c++) {

			if (row + r < 0 || row + r >= n_rows || column - 2 < 0 || column - 2 >= n_columns) continue;
			Sx += midput_data[(int) offset + (r * n_columns - 2)].x * filter_x_weights[filter_x_index++];
            Sy += midput_data[(int) offset + (r * n_columns - 2)].x * filter_y_weights[filter_y_index++];

			if (row + r < 0 || row + r >= n_rows || column - 1 < 0 || column - 1 >= n_columns) continue;
			Sx += midput_data[(int) offset + (r * n_columns - 1)].x * filter_x_weights[filter_x_index++];
            Sy += midput_data[(int) offset + (r * n_columns - 1)].x * filter_y_weights[filter_y_index++];

			if (row + r < 0 || row + r >= n_rows || column < 0 || column >= n_columns) continue;
			Sx += midput_data[(int) offset + (r * n_columns)].x * filter_x_weights[filter_x_index++];
            Sy += midput_data[(int) offset + (r * n_columns)].x * filter_y_weights[filter_y_index++];

			if (row + r < 0 || row + r >= n_rows || column + 1 < 0 || column + 1 >= n_columns) continue;
			Sx += midput_data[(int) offset + (r * n_columns + 1)].x * filter_x_weights[filter_x_index++];
            Sy += midput_data[(int) offset + (r * n_columns + 1)].x * filter_y_weights[filter_y_index++];

			if (row + r < 0 || row + r >= n_rows || column + 2 < 0 || column + 2 >= n_columns) continue;
			Sx += midput_data[(int) offset + (r * n_columns + 2)].x * filter_x_weights[filter_x_index++];
            Sy += midput_data[(int) offset + (r * n_columns + 2)].x * filter_y_weights[filter_y_index++];

			//}
	}
	uchar rgb = (uchar) sqrt( (double)(Sx * Sx) + (double)(Sy * Sy) ) / 8;

	output_data[offset].x = rgb;
	output_data[offset].y = rgb;
	output_data[offset].z = rgb;
	output_data[offset].w = input_data[offset].w;

//	printf("%u %u %u %u \n", output_data[offset].x, output_data[offset].y, output_data[offset].z, output_data[offset].w);
}


/*

	uchar intensity;
	int filter_x_index = 0;
	int filter_y_index = 0;
	for (int r = -HFS; r <= HFS; r++) {
			//for (int c = -HFS; c <= HFS; c++) {

			if (row + r < 0 || row + r >= n_rows || column - 2 < 0 || column - 2 >= n_columns) continue;
			intensity = (uchar)(input_data[offset + (r * n_columns - 2)].x * gray[0])
								+ (uchar)(input_data[offset + (r * n_columns - 2)].y * gray[1])
								+ (uchar)(input_data[offset + (r * n_columns - 2)].z * gray[2]);
			Sx += intensity * filter_x_weights[filter_x_index++];
            Sy += intensity * filter_y_weights[filter_y_index++];

			if (row + r < 0 || row + r >= n_rows || column - 1 < 0 || column - 1 >= n_columns) continue;
			intensity = (uchar)(input_data[offset + (r * n_columns - 1)].x * gray[0])
								+ (uchar)(input_data[offset + (r * n_columns - 1)].y * gray[1])
								+ (uchar)(input_data[offset + (r * n_columns - 1)].z * gray[2]);
			Sx += intensity * filter_x_weights[filter_x_index++];
            Sy += intensity * filter_y_weights[filter_y_index++];

			if (row + r < 0 || row + r >= n_rows || column < 0 || column >= n_columns) continue;
			intensity = (uchar)(input_data[offset + (r * n_columns)].x * gray[0])
								+ (uchar)(input_data[offset + (r * n_columns)].y * gray[1])
								+ (uchar)(input_data[offset + (r * n_columns)].z * gray[2]);
			Sx += intensity * filter_x_weights[filter_x_index++];
            Sy += intensity * filter_y_weights[filter_y_index++];

			if (row + r < 0 || row + r >= n_rows || column + 1 < 0 || column + 1 >= n_columns) continue;
			intensity = (uchar)(input_data[offset + (r * n_columns + 1)].x * gray[0])
								+ (uchar)(input_data[offset + (r * n_columns + 1)].y * gray[1])
								+ (uchar)(input_data[offset + (r * n_columns + 1)].z * gray[2]);
			Sx += intensity * filter_x_weights[filter_x_index++];
            Sy += intensity * filter_y_weights[filter_y_index++];

			if (row + r < 0 || row + r >= n_rows || column + 2 < 0 || column + 2 >= n_columns) continue;
			intensity = (uchar)(input_data[offset + (r * n_columns + 2)].x * gray[0])
								+ (uchar)(input_data[offset + (r * n_columns + 2)].y * gray[1])
								+ (uchar)(input_data[offset + (r * n_columns + 2)].z * gray[2]);
			Sx += intensity * filter_x_weights[filter_x_index++];
            Sy += intensity * filter_y_weights[filter_y_index++];

			//}
	}
	uchar rgb = (uchar) sqrt( (double)(Sx * Sx) + (double)(Sy * Sy) ) / 8;

*/