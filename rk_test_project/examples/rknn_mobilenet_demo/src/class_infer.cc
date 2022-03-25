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
#include <queue>
#include <string>
#include <vector>
#include <stdio.h>
#include <stdarg.h>
#include <sstream>
#include <sys/stat.h>
#include <unistd.h>
#include <dirent.h>
#include <unistd.h>
#include <sys/vfs.h>
// #include "opencv2/imgcodecs/legacy/constants_c.h"
// #include "opencv2/core/core.hpp"
// #include "opencv2/imgproc/imgproc.hpp"
// #include<opencv2/opencv.hpp>
// #include "opencv2/highgui/highgui.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "rknn_api.h"



using namespace std;
using namespace cv;

#define NUM_CLASS 34
#define NET_OUT_WIDTH 65
#define NET_OUT_HEIGHT 65

#define HAS_SOFT_MAX 0

/*-------------------------------------------
                  Functions
-------------------------------------------*/

// const string class_name[] = {"background","aeroplane","bicycle","bird","boat","bottle","bus","car",
//                             "cat","chair","cow","diningtable","dog","horse",
//                             "motorbike","person","pottedplant","sheep","sofa","train","tv"};

// const string class_name[] = {"cat","dog","pet_feces","shoe","slipper","socks","unknown"};
// const string class_name[] = {"blanket","tile_floor","wood_floor"};

//31cls
// const string class_name[] = {"bed","bed_1","blanket","cabinet","chair_base","charger","cupboard",
//                              "cupboard_1","dining_table","dirt","door_anno","invalid","metal_chair_foot","dog",
//                              "pet_feces","refrigerator","refrigerator_1","shoe","socks","sofa","sofa_1",
//                              "tea_table","tile_floor","toilet","TV_stand","unknown","washing_machine","washing_machine_1",
//                             "weight_scale","wire","wood_floor"};

//33cls
// const string class_name[] = {"bed","bed_1","blanket","cabinet","chair_base","charger","cupboard",
//                             "cupboard_1","dining_table","dirt","door_anno","invalid","metal_chair_foot","dog",
//                             "pet_feces","plastic_toy","refrigerator","refrigerator_1","shoe","socks","sofa","sofa_1",
//                             "tea_table","tile_floor","toilet","TV_stand","unknown","wallet","washing_machine","washing_machine_1",
//                             "weight_scale","wire","wood_floor"};


//34cls
const string class_name[] = {"bed","bed_1","blanket","cabinet","chair_base","charger","cupboard",
                            "cupboard_1","dining_table","dirt","door_anno","invalid","metal_chair_foot","person_leg","dog",
                            "pet_feces","plastic_toy","refrigerator","refrigerator_1","shoe","socks","sofa","sofa_1",
                            "tea_table","tile_floor","toilet","TV_stand","unknown","wallet","washing_machine","washing_machine_1",
                            "weight_scale","wire","wood_floor"};

const int img_width = 128;
const int img_height = 128;
const int img_channels = 3;

rknn_context ctx;
int ret;
int model_len = 0;
unsigned char *model;
// const char *model_path = "./yolo4_tiny_t.rknn";
rknn_input_output_num io_num;
rknn_tensor_attr input_attrs[100];
rknn_tensor_attr output_attrs[100];

double get_us(struct timeval t) { return (t.tv_sec * 1000000 + t.tv_usec); }

long time_diff(timeval start_time, timeval stop_time)
{
    return (get_us(stop_time) - get_us(start_time))/1000;
}

uint64_t time_tToTimestamp(const time_t &t ){
    return (((uint64_t)t) * (uint64_t)10000000) + ((uint64_t)116444736*1000000000);
}

uint64_t get_sys_time_interval(){
    timespec  tim;
    clock_gettime(CLOCK_MONOTONIC, &tim);
    return (time_tToTimestamp( tim.tv_sec ) + tim.tv_nsec/100)/10000;
}

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

// A sprintf-like function for std::string
static std::string evformat(const char *fmt, ...)
{
	if (!fmt) return std::string("");

	int   result = -1, length = 1024;
	std::vector<char> buffer;
	while (result == -1)
	{
		buffer.resize(length + 10);

		va_list args;  // This must be done WITHIN the loop
		va_start(args,fmt);
		result = vsnprintf(&buffer[0], length, fmt, args);
		va_end(args);

		// Truncated?
		if (result>=length) result=-1;
		length*=2;
	}
	std::string s(&buffer[0]);
	return s;
}

bool readDirFileList(const char *directory, std::vector<std::string> &files_list,std::vector<std::string> &files_name_list, bool recursion)
{
    DIR *dir;
    struct dirent *ptr;
    char base[1000];
    const char *base_path = directory;

    if ((dir=opendir(base_path)) == NULL)
    {
        printf("[CFileSystem]Can not open directory %s !\n", directory);
        return false;
    }

    while ((ptr=readdir(dir)) != NULL)
    {
        if(strcmp(ptr->d_name,".")==0 || strcmp(ptr->d_name,"..")==0)
        {
            /* Current dir OR parrent dir */
            continue;
        }
        else if(ptr->d_type == 8)
        {
            /* File */
            std::string str = evformat("%s/%s", base_path, ptr->d_name);
            std::string str_name = evformat("%s", ptr->d_name);
            std::cout << "ori str :" << str_name << std::endl;
            std::string name_str = str_name.replace(str_name.find(".jpg"), 4, "");
            std::cout << "after replaced str :" << name_str << std::endl;
            files_name_list.push_back(name_str);
            files_list.push_back(str);
        }
        else if(ptr->d_type == 10)
        {
            /* Link file */
        }
        else if(ptr->d_type == 4)
        {
            /* Directory */
            if(recursion)
            {
                memset(base,'\0',sizeof(base));
                strcpy(base, base_path);
                strcat(base,"/");
                strcat(base, ptr->d_name);
                readDirFileList(base, files_list,files_name_list, true);
            }
        }
    }
    closedir(dir);
    return true;
}

int init_class_model(const char *model_path)
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

void getTopN(float* prediction, int prediction_size, size_t num_results,
                                          float threshold, std::vector<std::pair<float, int>>* top_results,
                                          bool input_floating) 
{
    // Will contain top N results in ascending order.
    std::priority_queue<std::pair<float, int>, std::vector<std::pair<float, int>>,
                        std::greater<std::pair<float, int>>> top_result_pq;

    const long count = prediction_size;
   
    for (int i = 0; i < count; ++i) 
    {
        // std::cout << " i is " << i << " prediction is " << prediction[i] << std::endl;
        float value;
        if (input_floating)
        {
            value = prediction[i];
        }
        else
        {
            value = prediction[i] / 255.0;
        }   
      
        // Only add it if it beats the threshold and has a chance at being in
        // the top N.
        // if (value < m_ai_parameter.getAiObjectClassSorceThreshold(CTypeTransform::aiLable2AiObjectClass((AIAllClassLabel)i)))
        if (value < threshold)
        {
            continue;
        }

        top_result_pq.push(std::pair<float, int>(value, i));

        // If at capacity, kick the smallest value out.
        if (top_result_pq.size() > num_results) 
        {
            top_result_pq.pop();
        }
    }
 
    // Copy to output vector and reverse into descending order.
    while (!top_result_pq.empty()) 
    {
        top_results->push_back(top_result_pq.top());
        top_result_pq.pop();
    }

    std::reverse(top_results->begin(), top_results->end());
}

/***********************************************************************************
Function:     activation_function_softmax
Description:  None
Input:        None
Output:       None
Return:       None
Others:       None
***********************************************************************************/
void activation_function_softmax(float* src, float* dst, int length)
{
	float max = *std::max_element(src, src + length);
	float sum=0;
 
	for (int i = 0; i < length; ++i) {
		dst[i] = std::exp(src[i] - max);
		sum += dst[i];
	}
 
	for (int i = 0; i < length; ++i) {
		dst[i] /= sum;
        // printf("src = %.2f, dest = %.2f\n", src[i],dst[i]);
	}
}

pair<float, int> run_class_model(cv::Mat img)
{
    pair<float,int> default_result(0.00001,-1);
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
        return default_result;
    }

    // Run
    printf("rknn_run\n");
    ret = rknn_run(ctx, nullptr);
    if(ret < 0) {
        printf("rknn_run fail! ret=%d\n", ret);
        return default_result;
    }
    printf("rknn_run end\n");
    // Get Output
    rknn_output outputs[1];
    memset(outputs, 0, sizeof(outputs));
    outputs[0].want_float = 1;
    ret = rknn_outputs_get(ctx, 1, outputs, NULL);
    if(ret < 0) {
        printf("rknn_outputs_get fail! ret=%d\n", ret);
        return default_result;
    }
    // printf("rknn run spend time--------:%ldms\n",get_sys_time_interval() - end_init);

    float *out_buf = (float*)(outputs[0].buf);

    // for (int i = 0; i < NUM_CLASS; i++)
    // {
    //     printf("index:%d score :%.2f\n",i,out_buf[i]);
    // }
    long stop = get_sys_time_interval();

    std::vector<std::pair<float, int>> top_results;

    float *src_prediction = (float *)(outputs[0].buf);

    #if HAS_SOFT_MAX
    //softmaxs
    float *dest_prediction = new float[output_attrs[0].n_elems];
    memset(dest_prediction, 0, output_attrs[0].n_elems);
    activation_function_softmax(src_prediction,dest_prediction,output_attrs[0].n_elems);
    getTopN(dest_prediction, output_attrs[0].n_elems,
                  1, 0.001, &top_results, outputs[0].want_float);
    delete []dest_prediction;

    #else
    getTopN(src_prediction, output_attrs[0].n_elems,
                  1, 0.01, &top_results, outputs[0].want_float);
    #endif
 
    
    rknn_outputs_release(ctx, 1, outputs);

    // int class_num = top_results[0].second;
    // double classify_score = top_results[0].first;
    return top_results[0];

}

void destroy_class_model()
{
  // Release
    if(ctx >= 0) {
        rknn_destroy(ctx);
    }
    if(model) {
        free(model);
    }
}


void save_class_result(std::string detect_info_str)
{
    std::string debug_file_name = "/tmp/ai_mode_test_result.txt";
    // std::ofstream out_file(debug_file_name, ios::app);
    std::ofstream out_file(debug_file_name, ios::out | ios::app);
    out_file<<detect_info_str;
    out_file<<std::endl;
    out_file.close();
}


void processAutoTestModel()
{
    double all_class_rate[NUM_CLASS] = {0};
    int all_class_total_count[NUM_CLASS] = {0};
    int all_class_pre_true_count[NUM_CLASS] = {0};
    int per_class_precount[NUM_CLASS] = {0};
    for(int cur_cls_index = 0; cur_cls_index < NUM_CLASS; cur_cls_index++)
    {
        std::string cur_class_name = class_name[cur_cls_index];
        std::string directory = "/tmp/AI/mode_test/" + cur_class_name;
        std::vector<std::string> files_list;
        std::vector<std::string> files_name_list;
        if(readDirFileList(directory.c_str(), files_list,files_name_list, true))
        {
            printf("directory files_list size is %d %s!\n", files_list.size(),cur_class_name);

            int sucess_times = 0;
            int all_times = 0;
            double process_time = 0.0;
            double all_process_time = 0.0;
            double total_score = 0.0;
            int right = 0;
            for(size_t i = 0; i < files_list.size(); i++)
            {
                //执行模型，输出结果
                std::string name = files_list[i];
                std::string file_pre_name = files_name_list[i];

                struct timeval startread, startt, startr;
                struct timeval stopread, stoptt, stoptr;

                printf("------------------------------------------------");
                // long startread = get_sys_time_interval();
                gettimeofday(&startread, nullptr);
                // cv::Mat orig_img = cv::imread(name);
                cv::Mat orig_img = cv::imread(name, cv::IMREAD_COLOR);

                gettimeofday(&stopread, NULL);
                // printf("1806 read a small image time is --------:%ldms\n",get_sys_time_interval() - startread);
                printf("1806 read a small image time is --------:%ldms\n",time_diff(startread,stopread)); 
                cv::cvtColor(orig_img, orig_img,  cv::COLOR_BGR2RGB);
            
                cv::Mat img = orig_img.clone();
                // long startr = get_sys_time_interval();
                gettimeofday(&startr, nullptr);
                if(orig_img.cols != img_width || orig_img.rows != img_height) {
                    // printf("resize %d %d to %d %d\n", orig_img.cols, orig_img.rows, img_width, img_height);
                    cv::resize(orig_img, img, cv::Size(img_height, img_height), (0, 0), (0, 0), cv::INTER_LINEAR);
                }
                gettimeofday(&stoptr, NULL);
                // printf("resize time--------:%ldms\n",get_sys_time_interval() - startr);
                printf("resize time--------:%ldms\n",time_diff(startr,stoptr));
                // long startt = get_sys_time_interval();

                gettimeofday(&startt, nullptr);
                pair<float, int> pre_result = run_class_model(img);
                gettimeofday(&stoptt, NULL);
                // printf("rknn run spend time--------:%ldms\n",get_sys_time_interval() - startt);
                printf("rknn run spend time--------:%ldms\n",time_diff(startt,stoptt));
                int pre_class_num = pre_result.second;
                // int per_class_pre_count_now = per_class_precount[pre_class_num];
                // per_class_pre_count_now ++;
                per_class_precount[pre_class_num] += 1;
            
                if(cur_class_name == class_name[pre_class_num])
                {
                    sucess_times++;
                }
                all_times++;
                printf("ori_class is %s============%s is pred score:%.3f,cls:%d\n",cur_class_name.c_str(),class_name[pre_class_num].c_str(),pre_result.first,pre_class_num);

                std::string class_num_debug_str = to_string(pre_class_num);
                std::string class_score_debug_str = to_string(pre_result.first);
                std::string debug_info_str = name + " " + class_num_debug_str + " " + class_score_debug_str;
                // save_class_result(debug_info_str);

            }
            // double success_rate = (double)sucess_times / all_times;
            double success_rate = sucess_times * 1.0 / all_times;
            // double av_process_time = all_process_time / all_times;
            all_class_rate[cur_cls_index] = success_rate;
            all_class_total_count[cur_cls_index] = all_times;
            all_class_pre_true_count[cur_cls_index] = sucess_times;
            printf("class %s total_count %d pre_true_count %d recall %.3f!\n", class_name[cur_cls_index].c_str(),all_times,sucess_times,success_rate);
        }
  
    }

    int total_count = 0;
    int total_pre_true_count = 0;
    int total_pre_count = 0;
    
    for(size_t j = 0; j < NUM_CLASS; j++)
    {
        // if(all_class_total_count[j] == 0)
        //     continue;
        double recall_rate = all_class_rate[j];
        double precision_rate = all_class_pre_true_count[j] * 1.0 / per_class_precount[j];
        total_count += all_class_total_count[j];
        total_pre_true_count += all_class_pre_true_count[j];
        total_pre_count += per_class_precount[j];
        printf("%s %d %d %d %.3f %.3f\n", class_name[j].c_str(),all_class_pre_true_count[j],per_class_precount[j]-all_class_pre_true_count[j],all_class_total_count[j],precision_rate,recall_rate);
        // printf("class %s total_count %d pre_count %d pre_true_count %d precision %.3f recall %.3f\n", class_name[j].c_str(),all_class_total_count[j],per_class_precount[j],all_class_pre_true_count[j],precision_rate,recall_rate);
    }
    double precison = total_pre_true_count*1.0/total_pre_count;
    double recall = total_pre_true_count*1.0/total_count;
    printf("all_class precision %.3f recall %.3f total_count %d pre_count %d pre_true_count %d\n",precison,recall,total_count,total_pre_count,total_pre_true_count);

}

int main(int argc, char** argv)
{
    const char *model_path = argv[1];
    // const char *img_path = argv[2];
    init_class_model(model_path);
    processAutoTestModel();
}

// /*-------------------------------------------
//                   Main Function
// -------------------------------------------*/
// int main(int argc, char** argv)
// {
//     const int img_width = 128;
//     const int img_height = 128;
//     const int img_channels = 3;

//     rknn_context ctx;
//     int ret;
//     int model_len = 0;
//     unsigned char *model;

//     const char *model_path = argv[1];
//     const char *img_path = argv[2];
//      // unsigned long start_time,end_load_model_time, stop_time;
//     timeval start_time,end_load_model_time,end_init_time,end_run_time,end_process_time, stop_time;
//     gettimeofday(&start_time, nullptr);
//     // start_time = GetTickCount();
//     long startt = get_sys_time_interval();
//     // Load image
//     cv::Mat orig_img = cv::imread(img_path, 1);
//     cvtColor(orig_img,orig_img, cv::COLOR_BGR2RGB);

//     cv::Mat img = orig_img.clone();
//     if(!orig_img.data) {
//         printf("cv::imread %s fail!\n", img_path);
//         return -1;
//     }
//     if(orig_img.cols != img_width || orig_img.rows != img_height) {
//         printf("resize %d %d to %d %d\n", orig_img.cols, orig_img.rows, img_width, img_height);
//         cv::resize(orig_img, img, cv::Size(img_width, img_height), (0, 0), (0, 0), cv::INTER_LINEAR);
//     }

//     // Load RKNN Model
//     model = load_model(model_path, &model_len);
//     gettimeofday(&end_load_model_time, nullptr);
//     // end_load_model_time = GetTickCount();
//     long end_load_model = get_sys_time_interval();
//     printf("end load model time:%ldms\n",end_load_model);
//     ret = rknn_init(&ctx, model, model_len, 0);
//     gettimeofday(&end_init_time, nullptr);
//     // end_load_model_time = GetTickCount();
//     long end_init = get_sys_time_interval();
//     printf("end init model time:%ldms\n",end_init);
//     if(ret < 0) {
//         printf("rknn_init fail! ret=%d\n", ret);
//         return -1;
//     }

//     // Get Model Input Output Info
//     rknn_input_output_num io_num;
//     ret = rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
//     if (ret != RKNN_SUCC) {
//         printf("rknn_query fail! ret=%d\n", ret);
//         return -1;
//     }
//     printf("model input num: %d, output num: %d\n", io_num.n_input, io_num.n_output);

//     printf("input tensors:\n");
//     rknn_tensor_attr input_attrs[io_num.n_input];
//     memset(input_attrs, 0, sizeof(input_attrs));
//     for (int i = 0; i < io_num.n_input; i++) {
//         input_attrs[i].index = i;
//         ret = rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, &(input_attrs[i]), sizeof(rknn_tensor_attr));
//         if (ret != RKNN_SUCC) {
//             printf("rknn_query fail! ret=%d\n", ret);
//             return -1;
//         }
//         printRKNNTensor(&(input_attrs[i]));
//     }

//     printf("output tensors:\n");
//     rknn_tensor_attr output_attrs[io_num.n_output];
//     memset(output_attrs, 0, sizeof(output_attrs));
//     for (int i = 0; i < io_num.n_output; i++) {
//         output_attrs[i].index = i;
//         ret = rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &(output_attrs[i]), sizeof(rknn_tensor_attr));
//         if (ret != RKNN_SUCC) {
//             printf("rknn_query fail! ret=%d\n", ret);
//             return -1;
//         }
//         printRKNNTensor(&(output_attrs[i]));
//     }

//     // Set Input Data
//     rknn_input inputs[1];
//     memset(inputs, 0, sizeof(inputs));
//     inputs[0].index = 0;
//     inputs[0].type = RKNN_TENSOR_UINT8;
//     inputs[0].size = img.cols*img.rows*img.channels();
//     inputs[0].fmt = RKNN_TENSOR_NHWC;
//     inputs[0].buf = img.data;

//     ret = rknn_inputs_set(ctx, io_num.n_input, inputs);
//     if(ret < 0) {
//         printf("rknn_input_set fail! ret=%d\n", ret);
//         return -1;
//     }

//     // Run
//     printf("rknn_run\n");
//     ret = rknn_run(ctx, nullptr);
//     if(ret < 0) {
//         printf("rknn_run fail! ret=%d\n", ret);
//         return -1;
//     }
//     printf("rknn_run end\n");
//     // Get Output
//     rknn_output outputs[1];
//     memset(outputs, 0, sizeof(outputs));
//     outputs[0].want_float = 1;
//     ret = rknn_outputs_get(ctx, 1, outputs, NULL);
//     if(ret < 0) {
//         printf("rknn_outputs_get fail! ret=%d\n", ret);
//         return -1;
//     }
//     printf("rknn run spend time--------:%ldms\n",get_sys_time_interval() - end_init);

//     Mat megeImg2(NET_OUT_HEIGHT,NET_OUT_WIDTH,CV_8UC3);
//     float *out_buf = (float*)(outputs[0].buf);

//     for (int i = 0; i < NUM_CLASS; i++)
//     {
//         printf("index:%d score :%.2f\n",i,out_buf[i]);
//     }
//     long stop = get_sys_time_interval();
//     printf("class infer spend time--------:%ldms\n",stop - end_init);
//     // printf("end detect time:%lds\n",stop);
//     // Release rknn_outputs
//     rknn_outputs_release(ctx, 1, outputs);

//     // Release
//     if(ctx >= 0) {
//         rknn_destroy(ctx);
//     }
//     if(model) {
//         free(model);
//     }
//     return 0;
// }


