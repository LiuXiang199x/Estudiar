/****************************************************************************
*
*    Copyright (c) 2017 - 2018 by Rockchip Corp.  All rights reserved.
*
*    The material in this file is confidential and contains trade secrets
*    of Rockchip Corporation. This is proprietary information owned by
*    Rockchip Corporation. No part of this work may be disclosed,
*    reproduced, copied, transmitted, or used in any way for any purpose,
*    without the express written permission of Rockchip Corporation.
*
*****************************************************************************/
#ifndef YOLOV3_POST_PROCESS_H
#define YOLOV3_POST_PROCESS_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define YOLOV4 1

//net_input_img_size / 32
#define GRID0 11
//net_input_img_size / 16
#define GRID1 22
#define GRID2 0 //52

#define SPAN 3

// NUMCLASS + 5
// #define LISTSIZE 6
#define LISTSIZE 9
// #define OBJ_THRESH 0.3
// #define NMS_THRESH 0.5
#define OBJ_THRESH 0.10
#define NMS_THRESH 0.5
#define NUM_CLS 4
// #define MAX_BOX 500
#define MAX_BOX 100

#ifdef __cplusplus
extern "C"{
#endif
int yolov3_post_process(float* input0, float* input1, float* out_pos, float* out_prop, int* out_label);
#ifdef __cplusplus
}
#endif


#endif //YOLOV3_POST_PROCESS_H

