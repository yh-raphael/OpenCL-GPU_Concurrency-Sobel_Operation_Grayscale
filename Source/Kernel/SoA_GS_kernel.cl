//	120210400 조 용 현
//  SoA_GS_kernel.cl
//
//  Written for CSEG437_CSE5437
//  Department of Computer Science and Engineering
//  Copyright © 2021 Sogang University. All rights reserved.

__kernel void SoA_GS_kernel(
	const __global uchar* input_data_R, __global uchar* output_data_R,
    const __global uchar* input_data_G, __global uchar* output_data_G,
    const __global uchar* input_data_B, __global uchar* output_data_B,
    const __global uchar* input_data_A, __global uchar* output_data_A,
	int n_columns, int n_rows)                          // image width & height
{
    int column = get_global_id(0);
    int row = get_global_id(1);
    int i = (n_columns * row + column);

    uchar intensity = (uchar)(0.299f * input_data_R[i]  // R
            + 0.587f * input_data_G[i]                  // G
            + 0.114f * input_data_B[i]);                // B

    output_data_R[i] = intensity;
    output_data_G[i] = intensity;
    output_data_B[i] = intensity;
    output_data_A[i] = input_data_A[i];
}