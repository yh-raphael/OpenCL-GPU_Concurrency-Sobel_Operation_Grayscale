//	120210400 조 용 현
//  AoS_GS_kernel.cl
//
//  Written for CSEG437_CSE5437
//  Department of Computer Science and Engineering
//  Copyright © 2021 Sogang University. All rights reserved.

__kernel void AoS_GS_kernel(
	const __global uchar* input_data, __global uchar* output_data,
	int n_columns, int n_rows)							// image width & height
{
	int column = get_global_id(0);
	int row = get_global_id(1);
	int i = 4 * (row * n_columns + column);

    uchar intensity = (uchar)(0.299f * input_data[i]	// R
            + 0.587f * input_data[i+1]					// G
            + 0.114f * input_data[i+2]);				// B

    output_data[i] = intensity;
	output_data[i+1] = intensity;
	output_data[i+2] = intensity;
	output_data[i+3] = input_data[i+3];
}