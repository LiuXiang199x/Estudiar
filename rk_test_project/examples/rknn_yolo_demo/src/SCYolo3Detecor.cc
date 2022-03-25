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

/*-------------------------------------------
                Includes
-------------------------------------------*/
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <fstream>
#include <iostream>
#include <sys/time.h>
#include "rknn_api.h"
#include <math.h>

// #define YOLO_3I 0

#ifdef YOLO_3I
#include "SCYolov3_post_process_3head.h"
#else
#include "SCYolov3_post_process.h"
#endif


#include "SCYolo3Detecor.h"

using namespace std;
using namespace cv;

/*-------------------------------------------
        Macros and Variables
-------------------------------------------*/

// const string class_name[] = { "person", "bicycle", "car","motorbike ","aeroplane ","bus ","train","truck ",
//                             "boat","traffic light","fire hydrant","stop sign ","parking meter","bench","bird","cat","dog ",
//                             "horse ","sheep","cow","elephant",
//                             "bear","zebra ","giraffe","backpack","umbrella","handbag","tie","suitcase","frisbee","skis","snowboard","sports ball","kite",
//                             "baseball bat","baseball glove","skateboard","surfboard","tennis racket","bottle","wine glass","cup","fork","knife ",
//                             "spoon","bowl","banana","apple","sandwich","orange","broccoli","carrot","hot dog","pizza ","donut","cake","chair","sofa",
//                             "pottedplant","bed","diningtable","toilet ","tvmonitor","laptop ","mouse    ","remote ","keyboard ","cell phone","microwave ",
//                             "oven ","toaster","sink","refrigerator ","book","clock","vase","scissors ","teddy bear ","hair drier", "toothbrush "};
// const string class_name[] = {"obj"};
const string class_name[] = {"obj1","obj2","obj3","obj4"};
/*-------------------------------------------
                  Functions
-------------------------------------------*/
const int img_width = 416;
const int img_height = 416;
const int img_channels = 3;
rknn_context ctx;
int ret;
int model_len = 0;
unsigned char *model;
const char *model_path = "./yolo4_tiny_t.rknn";
rknn_input_output_num io_num;
// rknn_tensor_attr input_attrs[io_num.n_input];
// rknn_tensor_attr output_attrs[io_num.n_output];
rknn_tensor_attr input_attrs[1];
#ifdef YOLO_3I
rknn_tensor_attr output_attrs[3];
#else
rknn_tensor_attr output_attrs[2];
#endif

static void printRKNNTensor(rknn_tensor_attr *attr) {
    printf("index=%d name=%s n_dims=%d dims=[%d %d %d %d] n_elems=%d size=%d fmt=%d type=%d qnt_type=%d fl=%d zp=%d scale=%f\n", 
            attr->index, attr->name, attr->n_dims, attr->dims[3], attr->dims[2], attr->dims[1], attr->dims[0], 
            attr->n_elems, attr->size, 0, attr->type, attr->qnt_type, attr->fl, attr->zp, attr->scale);
}

static unsigned char *load_model(const char *filename, int *model_size)
{
    FILE *fp = fopen(filename, "rb");
    if(fp == nullptr) {
        printf("fopen %s fail!\n", filename);
        return NULL;
    }
    fseek(fp, 0, SEEK_END);
    int model_len = ftell(fp);
    unsigned char *model = (unsigned char*)malloc(model_len);
    fseek(fp, 0, SEEK_SET);
    if(model_len != fread(model, 1, model_len, fp)) {
        printf("fread %s fail!\n", filename);
        free(model);
        return NULL;
    }
    *model_size = model_len;
    if(fp) {
        fclose(fp);
    }
    return model;
}

int init_yolo_model()
{
    // Load RKNN Model
    model = load_model(model_path, &model_len);
    ret = rknn_init(&ctx, model, model_len, 0);
    if(ret < 0) {
        printf("rknn_init fail! ret=%d\n", ret);
        return -1;
    }

    // Get Model Input Output Info
    ret = rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
    if (ret != RKNN_SUCC) {
        printf("rknn_query fail! ret=%d\n", ret);
        return -1;
    }
    printf("model input num: %d, output num: %d\n", io_num.n_input, io_num.n_output);

    printf("input tensors:\n");
    memset(input_attrs, 0, sizeof(input_attrs));
    for (int i = 0; i < io_num.n_input; i++) {
        input_attrs[i].index = i;
        ret = rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, &(input_attrs[i]), sizeof(rknn_tensor_attr));
        if (ret != RKNN_SUCC) {
            printf("rknn_query fail! ret=%d\n", ret);
            return -1;
        }
        printRKNNTensor(&(input_attrs[i]));
    }

    printf("output tensors:\n");
    memset(output_attrs, 0, sizeof(output_attrs));
    for (int i = 0; i < io_num.n_output; i++) {
        output_attrs[i].index = i;
        ret = rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &(output_attrs[i]), sizeof(rknn_tensor_attr));
        if (ret != RKNN_SUCC) {
            printf("rknn_query fail! ret=%d\n", ret);
            return -1;
        }
        printRKNNTensor(&(output_attrs[i]));
    }
}

cv::Mat run_yolo_model(cv::Mat img, cv::Mat orig_img)
{
    // Set Input Data
    rknn_input inputs[1];
    memset(inputs, 0, sizeof(inputs));
    inputs[0].index = 0;
    inputs[0].type = RKNN_TENSOR_UINT8;
    inputs[0].size = img.cols*img.rows*img.channels();
    inputs[0].fmt = RKNN_TENSOR_NHWC;
    inputs[0].buf = img.data;

    ret = rknn_inputs_set(ctx, io_num.n_input, inputs);
    if(ret < 0) {
        printf("rknn_input_set fail! ret=%d\n", ret);
        return cv::Mat();
    }

    // Run
    printf("rknn_run\n");
    ret = rknn_run(ctx, nullptr);
    if(ret < 0) {
        printf("rknn_run fail! ret=%d\n", ret);
        return cv::Mat();
    }

    // Get Output
    #ifdef YOLO_3I
    rknn_output outputs[3];
    #else
    rknn_output outputs[2];
    #endif

    memset(outputs, 0, sizeof(outputs));
    outputs[0].want_float = 1;
    outputs[1].want_float = 1;
    outputs[2].want_float = 1;
    ret = rknn_outputs_get(ctx, io_num.n_output, outputs, NULL);
    if(ret < 0) {
        printf("rknn_outputs_get fail! ret=%d\n", ret);
        return cv::Mat();
    }

    float out_pos[500*4];
    float out_prop[500];
    int out_label[500];
    int obj_num = 0;

    // Post Process
    #ifdef YOLO_3I
    obj_num = yolov3_post_process((float *)(outputs[0].buf), (float *)(outputs[1].buf),(float *)(outputs[2].buf), out_pos, out_prop, out_label);
    #else
    obj_num = yolov3_post_process((float *)(outputs[0].buf), (float *)(outputs[1].buf), out_pos, out_prop, out_label);
    #endif

    // Release rknn_outputs
    rknn_outputs_release(ctx, 2, outputs);
    printf("obj number :%lds\n",obj_num);
    // Draw Objects
    for (int i = 0; i < obj_num; i++) {
        printf("%s @ (%f %f %f %f) %f\n",
                class_name[out_label[i]].c_str(),
                out_pos[4*i+0], out_pos[4*i+1], out_pos[4*i+2], out_pos[4*i+3],
                out_prop[i]);
        // printf("%s @ (%f %f %f %f) %f\n",
        //     class_name[0].c_str(),
        //     out_pos[4*i+0], out_pos[4*i+1], out_pos[4*i+2], out_pos[4*i+3],
        //     out_prop[i]);
        int x1 = out_pos[4*i+0] * orig_img.cols;
        int y1 = out_pos[4*i+1] * orig_img.rows;
        int x2 = out_pos[4*i+2] * orig_img.cols;
        int y2 = out_pos[4*i+3] * orig_img.rows;
      
        printf("(%d %d %d %d)\n",x1,y1,x2,y2);
        printf("(widht--%df height---%d)\n",orig_img.cols,orig_img.rows);
        float score = out_prop[i];
        string score_str = to_string(score);
        rectangle(orig_img, Point(x1, y1), Point(x2, y2), Scalar(255, 0, 0, 255), 3);
        putText(orig_img, class_name[out_label[i]].c_str(), Point(x1, y1 - 12), 1, 2, Scalar(0, 255, 0, 255));
        // putText(orig_img, class_name[0].c_str(), Point(x1, y1 - 12), 1, 2, Scalar(0, 255, 0, 255));
        putText(orig_img, score_str, Point(x1, y1 - 30), 1, 2, Scalar(0, 255, 0, 255));
    }

    // printf("detect spend time--------:%ldms\n",stop - end_init);
    // printf("end detect time:%lds\n",stop);
    // imwrite("./out.jpg", orig_img);
    if (obj_num > 0){
        return orig_img;
    }else
    {
        return cv::Mat();
    }
}

void destroy_yolo_model()
{
  // Release
    if(ctx >= 0) {
        rknn_destroy(ctx);
    }
    if(model) {
        free(model);
    }
}

/*-------------------------------------------
                  Main Function
-------------------------------------------*/
