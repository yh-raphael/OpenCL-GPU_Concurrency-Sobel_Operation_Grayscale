//	120210400 조 용 현
//  Kernel-Optimized.cl						// Vector4 type variable + Loop unrolling + Shared memory.
//
//  Written for CSEG437_CSE5437
//  Department of Computer Science and Engineering
//  Copyright © 2021 Sogang University. All rights reserved.

#define INPUT(x, y)				input_data[n_columns * (y) + (x)] 
#define SHARED_MEM(x, y)		shared_mem[SMW * (y) + (x)]

#define		FS_5		5	// Filter Size
#define		HFS_5		2	// Half Filter Size

/* Applied Verctor-Type Variable. */
__kernel void Kernel_Optimized(
	const __global uchar4* input_data, __global uchar4* output_data,
	int n_columns, int n_rows, __constant int* filter_x_weights, __constant int* filter_y_weights,
	int filter_x_width, int filter_y_width, __local uchar4* shared_mem)
{    
	int column = get_global_id(0);
	int row = get_global_id(1);

	int loc_column = get_local_id(0);
	int loc_row = get_local_id(1);
	int SMW = get_local_size(0) + 2 * HFS_5;	// Shared Memory Width

	float gray[3] = {0.299f, 0.587f, 0.114f};
	int offset;


	/* Applied Shared Memory (Local Memory, Local Data Share) Technique. */
//	SHARED_MEM(loc_column + HFS_5, loc_row + HFS_5) = INPUT(column, row);
	offset = row * n_columns + column;
	uchar intensity = (uchar)(input_data[offset].x * gray[0]) + (uchar)(input_data[offset].y * gray[1])
						+ (uchar)(input_data[offset].z * gray[2]);
	SHARED_MEM(loc_column + HFS_5, loc_row + HFS_5).x = intensity;
	SHARED_MEM(loc_column + HFS_5, loc_row + HFS_5).y = intensity;
	SHARED_MEM(loc_column + HFS_5, loc_row + HFS_5).z = intensity;
	SHARED_MEM(loc_column + HFS_5, loc_row + HFS_5).w = input_data[offset].w;

	int side_left = 0, side_right = 0;
	if (loc_column < HFS_5) {
//	 	SHARED_MEM(loc_column, loc_row + HFS_5) = INPUT(column - HFS_5, row);
		offset = row * n_columns + column - HFS_5;
		uchar intensity = (uchar)(input_data[offset].x * gray[0]) + (uchar)(input_data[offset].y * gray[1])
						+ (uchar)(input_data[offset].z * gray[2]);
		SHARED_MEM(loc_column, loc_row + HFS_5).x = intensity;
		SHARED_MEM(loc_column, loc_row + HFS_5).y = intensity;
		SHARED_MEM(loc_column, loc_row + HFS_5).z = intensity;
		SHARED_MEM(loc_column, loc_row + HFS_5).w = input_data[offset].w;

		side_left = 1;
	}
	else if (loc_column >= get_local_size(0) - HFS_5) {
//		SHARED_MEM(loc_column + 2 * HFS_5, loc_row + HFS_5) = INPUT(column + HFS_5, row);
		offset = row * n_columns + column + HFS_5;
		uchar intensity = (uchar)(input_data[offset].x * gray[0]) + (uchar)(input_data[offset].y * gray[1])
						+ (uchar)(input_data[offset].z * gray[2]);
		SHARED_MEM(loc_column + 2 * HFS_5, loc_row + HFS_5).x = intensity;
		SHARED_MEM(loc_column + 2 * HFS_5, loc_row + HFS_5).y = intensity;
		SHARED_MEM(loc_column + 2 * HFS_5, loc_row + HFS_5).z = intensity;
		SHARED_MEM(loc_column + 2 * HFS_5, loc_row + HFS_5).w = input_data[offset].w;

		side_right = 1;
	}

	if (loc_row < HFS_5) {
//		SHARED_MEM(loc_column + HFS_5, loc_row) = INPUT(column, row - HFS_5);
		offset = (row - HFS_5) * n_columns + column;
		uchar intensity = (uchar)(input_data[offset].x * gray[0]) + (uchar)(input_data[offset].y * gray[1])
						+ (uchar)(input_data[offset].z * gray[2]);
		SHARED_MEM(loc_column + HFS_5, loc_row).x = intensity;
		SHARED_MEM(loc_column + HFS_5, loc_row).y = intensity;
		SHARED_MEM(loc_column + HFS_5, loc_row).z = intensity;
		SHARED_MEM(loc_column + HFS_5, loc_row).w = input_data[offset].w;

		if (side_left == 1) {
//			SHARED_MEM(loc_column, loc_row) = INPUT(column - HFS_5, row - HFS_5);
			offset = (row - HFS_5) * n_columns + column - HFS_5;
			uchar intensity = (uchar)(input_data[offset].x * gray[0]) + (uchar)(input_data[offset].y * gray[1])
							+ (uchar)(input_data[offset].z * gray[2]);
			SHARED_MEM(loc_column, loc_row).x = intensity;
			SHARED_MEM(loc_column, loc_row).y = intensity;
			SHARED_MEM(loc_column, loc_row).z = intensity;
			SHARED_MEM(loc_column, loc_row).w = input_data[offset].w;
		}
		if (side_right == 1) {
//			SHARED_MEM(loc_column + 2 * HFS_5, loc_row) = INPUT(column + HFS_5, row - HFS_5);
			offset = (row - HFS_5) * n_columns + column + HFS_5;
			uchar intensity = (uchar)(input_data[offset].x * gray[0]) + (uchar)(input_data[offset].y * gray[1])
							+ (uchar)(input_data[offset].z * gray[2]);
			SHARED_MEM(loc_column + 2 * HFS_5, loc_row).x = intensity;
			SHARED_MEM(loc_column + 2 * HFS_5, loc_row).y = intensity;
			SHARED_MEM(loc_column + 2 * HFS_5, loc_row).z = intensity;
			SHARED_MEM(loc_column + 2 * HFS_5, loc_row).w = input_data[offset].w;
		}
	}
	else if (loc_row >= get_local_size(1) - HFS_5) {
//		SHARED_MEM(loc_column + HFS_5, loc_row + 2 * HFS_5) = INPUT(column, row + HFS_5);
		offset = (row + HFS_5) * n_columns + column;
		uchar intensity = (uchar)(input_data[offset].x * gray[0]) + (uchar)(input_data[offset].y * gray[1])
						+ (uchar)(input_data[offset].z * gray[2]);
		SHARED_MEM(loc_column + HFS_5, loc_row + 2 * HFS_5).x = intensity;
		SHARED_MEM(loc_column + HFS_5, loc_row + 2 * HFS_5).y = intensity;
		SHARED_MEM(loc_column + HFS_5, loc_row + 2 * HFS_5).z = intensity;
		SHARED_MEM(loc_column + HFS_5, loc_row + 2 * HFS_5).w = input_data[offset].w;

		if (side_left == 1) {
//			SHARED_MEM(loc_column, loc_row + 2 * HFS_5) = INPUT(column - HFS_5, row + HFS_5);
			offset = (row + HFS_5) * n_columns + column - HFS_5;
			uchar intensity = (uchar)(input_data[offset].x * gray[0]) + (uchar)(input_data[offset].y * gray[1])
							+ (uchar)(input_data[offset].z * gray[2]);
			SHARED_MEM(loc_column, loc_row + 2 * HFS_5).x = intensity;
			SHARED_MEM(loc_column, loc_row + 2 * HFS_5).y = intensity;
			SHARED_MEM(loc_column, loc_row + 2 * HFS_5).z = intensity;
			SHARED_MEM(loc_column, loc_row + 2 * HFS_5).w = input_data[offset].w;
		}
		if (side_right == 1) {
//			SHARED_MEM(loc_column + 2 * HFS_5, loc_row + 2 * HFS_5) = INPUT(column + HFS_5, row + HFS_5);
			offset = (row + HFS_5) * n_columns + column + HFS_5;
			uchar intensity = (uchar)(input_data[offset].x * gray[0]) + (uchar)(input_data[offset].y * gray[1])
							+ (uchar)(input_data[offset].z * gray[2]);
			SHARED_MEM(loc_column + 2 * HFS_5, loc_row + 2 * HFS_5).x = intensity;
			SHARED_MEM(loc_column + 2 * HFS_5, loc_row + 2 * HFS_5).y = intensity;
			SHARED_MEM(loc_column + 2 * HFS_5, loc_row + 2 * HFS_5).z = intensity;
			SHARED_MEM(loc_column + 2 * HFS_5, loc_row + 2 * HFS_5).w = input_data[offset].w;
		}
	}

	barrier(CLK_LOCAL_MEM_FENCE);

	int Sx = 0;
	int Sy = 0;

	int filter_x_index = 0;
	int filter_y_index = 0;
	for (int r = -HFS_5; r <= HFS_5; r++) {
		
			/* Applied Loop Unrolling Technique. */
			if (row + r < 0 || row + r >= n_rows || column - 2 < 0 || column - 2 >= n_columns) continue;
			Sx += SHARED_MEM(loc_column + 0, loc_row + HFS_5 + r).x * filter_x_weights[filter_x_index++];
            Sy += SHARED_MEM(loc_column + 0, loc_row + HFS_5 + r).x * filter_y_weights[filter_y_index++];

			if (row + r < 0 || row + r >= n_rows || column - 1 < 0 || column - 1 >= n_columns) continue;
			Sx += SHARED_MEM(loc_column + 1, loc_row + HFS_5 + r).x * filter_x_weights[filter_x_index++];
            Sy += SHARED_MEM(loc_column + 1, loc_row + HFS_5 + r).x * filter_y_weights[filter_y_index++];

			if (row + r < 0 || row + r >= n_rows || column < 0 || column >= n_columns) continue;
			Sx += SHARED_MEM(loc_column + 2, loc_row + HFS_5 + r).x * filter_x_weights[filter_x_index++];
            Sy += SHARED_MEM(loc_column + 2, loc_row + HFS_5 + r).x * filter_y_weights[filter_y_index++];

			if (row + r < 0 || row + r >= n_rows || column + 1 < 0 || column + 1 >= n_columns) continue;
			Sx += SHARED_MEM(loc_column + 3, loc_row + HFS_5 + r).x * filter_x_weights[filter_x_index++];
            Sy += SHARED_MEM(loc_column + 3, loc_row + HFS_5 + r).x * filter_y_weights[filter_y_index++];

			if (row + r < 0 || row + r >= n_rows || column + 2 < 0 || column + 2 >= n_columns) continue;
			Sx += SHARED_MEM(loc_column + 4, loc_row + HFS_5 + r).x * filter_x_weights[filter_x_index++];
            Sy += SHARED_MEM(loc_column + 4, loc_row + HFS_5 + r).x * filter_y_weights[filter_y_index++];

	}
	uchar rgb = (uchar) sqrt( (double)(Sx * Sx) + (double)(Sy * Sy) ) / 8;

	offset = row * n_columns + column;
	output_data[offset].x = rgb;
	output_data[offset].y = rgb;
	output_data[offset].z = rgb;
	output_data[offset].w = input_data[offset].w;

//	printf("%u %u %u %u \n", output_data[offset].x, output_data[offset].y, output_data[offset].z, output_data[offset].w);
}