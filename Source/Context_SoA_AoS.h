//  120210400 조 용 현
//  Context_SoA_AoS.h
//
//  Written for CSEG437_CSE5437
//  Department of Computer Science and Engineering
//  Copyright © 2021 Sogang University. All rights reserved.
//

#ifndef __CONTEXT_SOA_AOS__
#define __CONTEXT_SOA_AOS__

#include <CL/cl.h>
#include <CL/cl_gl.h>

#define     QUEUED_TO_END         0
#define     SUBMIT_TO_END          1
#define     START_TO_END            2

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <windows.h>

#include <FreeImage/FreeImage.h>

#include "Util/my_OpenCL_util_2_2.h"

// [HW3]
#include "Config_SoA_AoS.h"

// Array of Structure를 위한 자료구조.
typedef struct _Pixel_Channels {
    BYTE R, G, B, A;
} Pixel_Channels;

// Structure of Array를 위한 자료구조.
typedef struct _Pixel_Planes {
    BYTE *R_plane, *G_plane, *B_plane, *A_plane;
} Pixel_Planes;

typedef struct _OPENCL_C_PROG_SRC {
    size_t length;
    char* string;
} OPENCL_C_PROG_SRC;

typedef struct _Context {
    FREE_IMAGE_FORMAT image_format;
    unsigned int image_width, image_height, image_pitch;
    size_t image_data_bytes;

    struct {
        int width;
        const int* weights;
    } sobel_filter_x;

    struct {
        int width;
        const int* weights;
    } sobel_filter_y;

    struct {
        FIBITMAP* fi_bitmap_32;
        BYTE* image_data;
    } input;
    struct {
        FIBITMAP* fi_bitmap_32;
        BYTE* image_data;
    } output;

    Pixel_Channels *AoS_image_input, *AoS_image_output, *AoS_image_midput;
    Pixel_Planes SoA_image_input, SoA_image_output, SoA_image_midput;

    ////////////////////////////// OpenCL related /////////////////////////////////
    cl_platform_id platform_id;
    cl_device_id device_id;
    cl_context context;
    cl_command_queue cmd_queue;
    cl_program program;
    cl_kernel kernel;                   // 커널 인스턴스 정보.
    OPENCL_C_PROG_SRC prog_src;
    cl_mem BO_input, BO_midput, BO_output;
    cl_mem BO_input_R,BO_midput_R,BO_output_R, BO_input_G,BO_midput_G,BO_output_G, BO_input_B,BO_midput_B,BO_output_B, BO_input_A,BO_midput_A,BO_output_A;
    cl_mem BO_filter_x, BO_filter_y;
    cl_event event_for_timing;

    cl_uint work_dim;
    size_t global_work_offset[3], global_work_size[3], local_work_size[3];

    // [HW3] Concurrency.
    cl_command_queue _cmd_queue_[MAXIMUM_COMMAND_QUEUES];
    cl_kernel _kernel_[MAXIMUM_COMMAND_QUEUES];

    cl_mem BO_input_dev, BO_midput_dev, BO_output_dev, BO_filter_x_dev, BO_filter_y_dev;
    cl_mem BO_input_pinned, BO_midput_pinned, BO_output_pinned, BO_filter_x_pinned, BO_filter_y_pinned;
//    Pixel_Channels *data_input, *data_midput, *data_output;
//    int *data_filter_x, *data_filter_y;

    int n_elements;
    size_t buffer_size_in_bytes;

    cl_event event_write_A[MAXIMUM_COMMAND_QUEUES];
    cl_event event_write_B[MAXIMUM_COMMAND_QUEUES];
    cl_event event_compute[MAXIMUM_COMMAND_QUEUES];
    cl_event event_read_C[MAXIMUM_COMMAND_QUEUES];

    Pixel_Channels* solution;
    int copy_compute_type;

} Context;

extern Context context;

// [HW2] GPU 커널 최적화.
static const int Sobel_x[25] = {
    -5, -4,  0,  4,  5,
    -8, -10, 0,  10, 8,
    -10,-20, 0,  20, 10,
    -8, -10, 0,  10, 8,
    -5,  -4, 0,  4,  5
};

static const int Sobel_y[25] = {
    -5,  -8, -10, -8, -5,
    -4, -10, -20,-10, -4,
     0,   0,   0,  0,  0,
     4,  10,  20, 10,  4,
     5,   8,  10,  8,  5
};

////////////////////// Image_IO.cpp /////////////////////////////////////////
void read_input_image_from_file32(const char* filename);
void prepare_output_image(void);
void write_output_image_to_file32(const char* filename);

void prepare_SoA_input_and_output(void);
void prepare_AoS_input_and_output(void);
void prepare_SoA_input_and_output_SO(void);
void prepare_AoS_input_and_output_SO(void);


void convert_SoA_output_to_output_image_data(void);
void convert_AoS_output_to_output_image_data(void);


///////////// My_Image_Filtering_Codes.cpp ///////////////////////////////////
void convert_to_greyscale_image_SoA_CPU(void);
void convert_to_greyscale_image_AoS_CPU(void);

///////////// OpenCL related codes //////////////////////////////
void apply_sobel_operator_to_image_SoA_CPU(void);
void apply_sobel_operator_to_image_AoS_CPU(void);

int initialize_OpenCL_AoS(void);
int initialize_OpenCL_SoA(void);
int initialize_OpenCL_SoA_SO(void);
int initialize_OpenCL_AoS_SO(void);

int set_local_work_size_and_kernel_arguments_AoS(void);
int set_local_work_size_and_kernel_arguments_SoA(void);
int set_local_work_size_and_kernel_arguments_SoA_SO(void);
int set_local_work_size_and_kernel_arguments_AoS_SO(void);

// [HW2] GPU 커널 최적화.
int set_local_work_size_and_kernel_arguments_AoS_SO_KO(void);           // Kernel_Naive, Kernel_Advanced.
int set_local_work_size_and_kernel_arguments_AoS_SO_KO_local(void);     // Kernel_Optimized.


int run_OpenCL_kernel_AoS(void);
int run_OpenCL_kernel_SoA(void);
int run_OpenCL_kernel_SoA_SO(void);
int run_OpenCL_kernel_AoS_SO(void);

void clean_up_system(void);

// [HW3]
void use_multiple_segments_and_three_command_queues_with_events_breadth(int n_segments);

#endif // __CONTEXT_SOA_AOS__