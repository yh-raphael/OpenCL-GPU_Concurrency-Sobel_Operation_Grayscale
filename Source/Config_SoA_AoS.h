//  120210400 조 용 현
//  Config_SoA_AoS.h
//
//  Written for CSEG437_CSE5437
//  Department of Computer Science and Engineering
//  Copyright © 2021 Sogang University. All rights reserved.
//

#ifndef Config_SoA_AoS_h
#define Config_SoA_AoS_h

/////////////////////////////////////////////////////////////////////////////////////
// GPU에서 돌릴 Kernel program 종류 enumerate.
#define		SoA_GS_CPU		0		// SoA Grayscale on CPU
#define		AoS_GS_CPU		1		// AoS Grayscale on CPU
#define		SoA_SO_CPU		2		// SoA Sobel operator on CPU
#define		AoS_SO_CPU		3		// AoS Sobel operator on CPU
#define		SoA_GS_GPU		4		// SoA Grayscale on GPU
#define		AoS_GS_GPU		5		// AoS Grayscale on GPU
#define		SoA_SO_GPU		6		// SoA Sobel operator on GPU
#define		AoS_SO_GPU		7		// AoS Sobel operator on GPU

// [HW2] GPU 커널 최적화.
#define		Kernel_Naive		8	// Not optimized version.
#define		Kernel_Advanced		9	// Vector type + Loop Unrolling optimization.
#define		Kernel_Optimized	10	// Vector type + Loop Unrolling + Shared Memory optimization.
#define		Three_Queues		11	// Sol2


// Input file로 사용할 이미지 file 이름.
#define INPUT_FILE_0			"Image_0_7360_4832"
#define INPUT_FILE_1			"Image_1_9984_6400"
#define INPUT_FILE_2			"Image_2_7680_4320"
#define INPUT_FILE_3			"Image_3_8960_5408"
#define INPUT_FILE_4			"Image_4_6304_4192"
#define INPUT_FILE_5			"Image_5_1856_1376"
#define INPUT_FILE_8			"Grass_texture_2048_2048"
#define INPUT_FILE_9			"Tiger_texture_512_512"
#define INPUT_FILE_10		"Plain_color_128_192_64_1024_1024"
/////////////////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////////////////
#define		INPUT_IMAGE						2
#define		IMAGE_OPERATION				Three_Queues
/////////////////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////////////////
#if INPUT_IMAGE == 0
#define	INPUT_FILE 			INPUT_FILE_0
#elif INPUT_IMAGE == 1
#define	INPUT_FILE 			INPUT_FILE_1
#elif INPUT_IMAGE == 2
#define	INPUT_FILE 			INPUT_FILE_2
#elif INPUT_IMAGE == 3
#define	INPUT_FILE 			INPUT_FILE_3
#elif INPUT_IMAGE == 4
#define	INPUT_FILE 			INPUT_FILE_4
#elif INPUT_IMAGE == 5
#define	INPUT_FILE 			INPUT_FILE_5
#elif INPUT_IMAGE == 8
#define	INPUT_FILE 			INPUT_FILE_8
#elif INPUT_IMAGE == 9
#define	INPUT_FILE 			INPUT_FILE_9
#else INPUT_IMAGE == 10
#define	INPUT_FILE 			INPUT_FILE_10
#endif

#define	INPUT_FILE_NAME			"Data/Input/" INPUT_FILE ".jpg"
#define	OUTPUT_FILE_NAME		"Data/Output/" INPUT_FILE "_out.png"
/////////////////////////////////////////////////////////////////////////////////////

#define		KERNEL_SELECTION			IMAGE_OPERATION	
#if   KERNEL_SELECTION == 0
#define OPENCL_C_PROG_FILE_NAME "Source/Kernel/SoA_GS_kernel.cl"
#define KERNEL_NAME "SoA_GS_kernel"
#elif KERNEL_SELECTION == 1
#define OPENCL_C_PROG_FILE_NAME "Source/Kernel/AoS_GS_kernel.cl"
#define KERNEL_NAME "AoS_GS_kernel"
#elif KERNEL_SELECTION == 2
#define OPENCL_C_PROG_FILE_NAME "Source/Kernel/SoA_SO_kernel.cl"
#define KERNEL_NAME "SoA_SO_kernel"
#elif KERNEL_SELECTION == 3
#define OPENCL_C_PROG_FILE_NAME "Source/Kernel/AoS_SO_kernel.cl"
#define KERNEL_NAME "AoS_SO_kernel"
#elif   KERNEL_SELECTION == 4
#define OPENCL_C_PROG_FILE_NAME "Source/Kernel/SoA_GS_kernel.cl"
#define KERNEL_NAME "SoA_GS_kernel"
#elif KERNEL_SELECTION == 5
#define OPENCL_C_PROG_FILE_NAME "Source/Kernel/AoS_GS_kernel.cl"
#define KERNEL_NAME "AoS_GS_kernel"
#elif KERNEL_SELECTION == 6
#define OPENCL_C_PROG_FILE_NAME "Source/Kernel/SoA_SO_kernel.cl"
#define KERNEL_NAME "SoA_SO_kernel"
#elif KERNEL_SELECTION == 7
#define OPENCL_C_PROG_FILE_NAME "Source/Kernel/AoS_SO_kernel.cl"
#define KERNEL_NAME "AoS_SO_kernel"

// [HW2] GPU 커널 최적화.
#elif KERNEL_SELECTION == 8
#define OPENCL_C_PROG_FILE_NAME "Source/Kernel/Kernel-Naive.cl"
#define KERNEL_NAME "Kernel_Naive"
#elif KERNEL_SELECTION == 9
#define OPENCL_C_PROG_FILE_NAME "Source/Kernel/Kernel-Advanced.cl"
#define KERNEL_NAME "Kernel_Advanced"
#elif KERNEL_SELECTION == 10
#define OPENCL_C_PROG_FILE_NAME "Source/Kernel/Kernel-Optimized.cl"
#define KERNEL_NAME "Kernel_Optimized"

// [HW3] Concurrency.
#elif KERNEL_SELECTION == 11
#define OPENCL_C_PROG_FILE_NAME "Source/Kernel/Three_Queues.cl"
#define KERNEL_NAME "Three_Queues"
#endif

#define		STAT_FILE_NAME				"Data/stat_file_KO.txt"
#define		N_EXECUTIONS				20

#define		LOCAL_WORK_SIZE_0			32		// Dim 0 (x)
#define		LOCAL_WORK_SIZE_1			8		// Dim 1 (y)

#define		SOBEL_FILTER_SIZE			5		// 5

#define MAKE_STAT_ITEM_LIST_CBO(string, gws, lws) sprintf((string), "IMAGE = %d(%s), GF_SIZE = %d, N_EXE = %d,\n " \
                    "    KER = %d, GWS = (%d, %d), LWS = (%d, %d)", INPUT_IMAGE, INPUT_FILE_NAME, GAUSSIAN_FILTER_SIZE, \
					N_EXECUTIONS, KERNEL_SELECTION, (gws)[0], (gws)[1], (lws)[0], (lws)[1]);

// [HW3] Concurrency.
#define		MAXIMUM_COMMAND_QUEUES		32


#endif