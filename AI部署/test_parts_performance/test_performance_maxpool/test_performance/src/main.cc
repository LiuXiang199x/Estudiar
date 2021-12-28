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
#include <vector>
#include <string.h>
#include <uchar.h>
#include <numeric>

#include "rknn_api.h"

using namespace std;
//using namespace cv;

#define BATCH_SIZE 1
#define uchar unsigned char

#define img_width 64
#define img_height 64
#define img_channels 3

double min_range = 0.5 - 0.117;
double max_range = 0.5 + 0.059;

vector<vector<vector<double>>> processTarget(vector<vector<double>> map_data, int idx, int idy);
int predictions(vector<vector<vector<double>>> inputs, int expand_type, vector<vector<double>> mask, int &res_idx, int &res_idy);
vector<vector<double>> crop_map(vector<vector<double>> tmp, int x, int y, double padding_num);
vector<vector<double>> get_frontier(vector<vector<double>> explored_map, vector<vector<double>> occupancy_map, int row, int column);
vector<vector<double>> filter_frontier(vector<vector<double>> map, int row, int column, int minimum_size);
void pro_target(vector<float> outputs, int expand_type, vector<vector<double>> mask, int &res_idx, int &res_idy);

/*-------------------------------------------
                  Functions
				  input_maps = [global_map, visited_map, agent_position_onehot]     +      output * frontier_mask
-------------------------------------------*/


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


class MaxPolling {
public:

	// 最大池化函数
	template <typename _Tp>
	_Tp* poll(_Tp* matrix, int matrix_w, int matrix_h, int kernel_size, int stride, bool show) {

		// 池化结果的size
		int result_w = (matrix_w - kernel_size) / stride + 1, result_h = (matrix_h - kernel_size) / stride + 1;
		// 申请内存
		_Tp* result = (_Tp*)malloc(sizeof(_Tp) * result_w * result_h);

		int x = 0, y = 0;
		for (int i = 0; i < result_h; i++) {
			for (int j = 0; j < result_w; j++) {
				result[y * result_w + x] = getMax(matrix, matrix_w, matrix_h, kernel_size, j * stride, i * stride);
				x++;
			}
			y++; x = 0;
		}
		if (show) {
			showMatrix(result, result_w, result_h);
		}
		return result;
	}

	template <typename _Tp>
	void showMatrix(_Tp matrix, int matrix_w, int matrix_h) {
		for (int i = 0; i < matrix_h; i++) {
			for (int j = 0; j < matrix_w; j++) {
				std::cout << matrix[i * matrix_w + j] << " ";
			}
			std::cout << std::endl;
		}
	}

	// 取kernel中最大值
	template <typename _Tp>
	_Tp getMax(_Tp* matrix, int matrix_w, int matrix_h, int kernel_size, int x, int y) {
		int max_value = 0;
		max_value = accumulate(matrix + matrix_w * (y + 0) + x + 0, matrix + matrix_w * (y + 0) + x + kernel_size, 0) + accumulate(matrix + matrix_w * (y + 1) + x + 0, matrix + matrix_w * (y + 1) + x + kernel_size, 0) + accumulate(matrix + matrix_w * (y + 2) + x + 0, matrix + matrix_w * (y + 2) + x + kernel_size, 0) + accumulate(matrix + matrix_w * (y + 3) + x + 0, matrix + matrix_w * (y + 3) + x + kernel_size, 0) + accumulate(matrix + matrix_w * (y + 4) + x + 0, matrix + matrix_w * (y + 4) + x + kernel_size, 0);
		/*
		for (int i = 0; i < kernel_size; i++) {
			for (int j = 0; j < kernel_size; j++) {
				if (max_value < matrix[matrix_w * (y + i) + x + j]) {
					max_value = matrix[matrix_w * (y + i) + x + j];
				}
			}
		}
		*/
		if(max_value==0){
			max_value = 0;
		}
		else{
			max_value = 1;
		}
		return max_value;
	}

	void testMaxPolling() {
		int matrix[36] = { 1,3,1,3,5,1,4,7,5,7,9,12,1,4,6,2,5,8,6,3,9,2,1,5,8,9,2,4,6,8,4,12,54,8,0,23 };
		poll(matrix, 6, 6, 2, 2, true);
	}
};

vector<vector<double>> crop_map(vector<vector<double>> tmp, int x, int y, double padding_num) {
	vector<vector<double>> map(1200 + 240, vector<double>(1200 + 240, padding_num));
	vector<vector<double>> map_output(240, vector<double>(240));
	int robot_x = x + 120;
	int robot_y = y + 120;

	for (int i = 0; i < 1200; i++) {
		for (int j = 0; j < 1200; j++) {
			map[120 + i][120 + j] = tmp[i][j];
		}
	}

	for (int i = 0; i < 240; i++) {
		for (int j = 0; j < 240; j++) {
			map_output[i][j] = map[robot_x - 120 + i][robot_y - 120 + j];
		}
	}
	return map_output;
}


vector<vector<vector<double>>> processTarget(vector<vector<double>> map_data, int idx, int idy) {

	// static vector<vector<double>> map(800, vector<double>(800, 0));
	printf("======== Start processing datas =======");
	int size_x = 800;
	int size_y = 800;
	// int size_x = maps_original.getSizeX();
	// int size_y = maps_original.getSizeY();
	int x_origin = 200;
	int y_origin = 200;
	double tmp_value;
	int robot__x;
	int robot__y;
	int expand_type;


	// global map[0] = map occupancy: -1/100-->1(unexplored space/obstacles); 0-->0(free space) --- expand with 1
	// global map[1] = explored states: 0/100-->1(free space/obstacles); -1-->0(unexplored space) --- expand with 0
	// global pose = agent status on the map: 1-->robot position; 0-->No --- expand with 0
	// frontier map: 1-->frontiers; 0-->not frontiers --- expand with 0 before FBE: 1:explored spaces, 0:unexplored spaces
	// obstacles <-- 0.5 --> free space: threshold = [min_range, max_range]

	static vector<vector<int>> map_occupancy(1200, vector<int>(1200, 1));
	static vector<vector<int>> explored_states(1200, vector<int>(1200, 0));
	static vector<vector<int>> agent_status(1200, vector<int>(1200, 0));

	int* Ocp_pooling;
	int* Expp_pooling;
	int* Agentp_pooling;
	int* Frontp_pooling;

	static vector<vector<int>> Ocp_crop(240, vector<int>(240));
	static vector<vector<int>> Expp_crop(240, vector<int>(240));
	static vector<vector<int>> Visitedmap_crop(240, vector<int>(240));
	static vector<vector<int>> Agentp_crop(240, vector<int>(240));

	static int Ocmap[1440000];
	static int Expmap[1440000] = {0};
	static int Agentmap[1200 * 1200] = {0};
	static int Frontmap[1200 * 1200] = {0};
	memset(Ocmap, 1, sizeof(int)*1440000);
	
	printf("======== Getting maps data =======\n");

	printf("\n");


	printf("======== Pooling & Cropping maps =======\n");
	// timeval start_maxpoolmaps, end_maxpoolmaps, start_cropmaps, end_cropmaps;

	printf("======== Getting pooling maps =======\n");
	// long start_maxpoolmaps = get_sys_time_interval();


	MaxPolling pool2d;
	// maps:(1200, 1200) ---> (240, 240)
	Ocp_pooling = pool2d.poll(Ocmap, 1200, 1200, 5, 5, false);
	Expp_pooling = pool2d.poll(Expmap, 1200, 1200, 5, 5, false);
	Agentp_pooling = pool2d.poll(Agentmap, 1200, 1200, 5, 5, false);
	Frontp_pooling = pool2d.poll(Frontmap, 1200, 1200, 5, 5, false);

	vector<vector<vector<double>>> mappp;
	return mappp;
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

vector<vector<double>> get_inputs(int &robotx, int &roboty){
    // srand(time(0));
    static vector<vector<double>> inputs(800, vector<double>(800));

    for(int i=0; i<800; i++){
		for(int j=0;j<800;j++){
			inputs[i][j] = rand() * 1.0 / RAND_MAX;
		}
	}

	int robot_x = rand()%100+300;
	int robot_y = rand()%100+200;
	inputs[robot_x][robot_y] = 1;
	robotx = robot_x;
	roboty = robot_y;
	printf("====>robot_x=%d, robot_y=%d<====\n", robot_x, robot_y);
	printf("====> 800*800 random maps generated !!! <====\n");

    return inputs;
}


/*-------------------------------------------
                  Main Function
-------------------------------------------*/
int main(int argc, char** argv)
{
    // const int img_width = 224;
    // const int img_height = 224;
    // const int img_channels = 3;

    rknn_context ctx;
    int ret;
    int model_len = 0;
    unsigned char *model;
    srand((unsigned)time(NULL));

    const char *model_path = "../model/ckpt_precompile_20.rknn";
    const char *img_path = argv[2];

	timeval startt_getdata, end_getdata, start_ProcessData, end_ProcessData;
	for(int counter_test=0; counter_test<1000; counter_test++){
		// ================ GET RANDOM INPUT DATA ==================
		int robot_xx, robot_yy;
		long startt_getdata = get_sys_time_interval();
		static vector<vector<double>> data_vectorrr;
		static vector<vector<vector<double>>> data_vector;
		data_vectorrr = get_inputs(robot_xx, robot_yy);
		long end_getdata = get_sys_time_interval();
		printf("====================> robot_xx = %d, robot_yy = %d\n", robot_xx, robot_yy);
		printf("get random input data--------:%ldms\n",end_getdata - startt_getdata);
		printf("\n");
		// =========================================================


		// ================ Processing all datas ==================
		long start_ProcessData = get_sys_time_interval();
		data_vector = processTarget(data_vectorrr, robot_xx, robot_yy);
		long end_ProcessData = get_sys_time_interval();
		printf("Processing datas --------:%ldms\n",end_ProcessData - start_ProcessData);
		printf("\n");
		// ===================================================
		/*
		// uchar batch_img_data[img.cols*img.rows*img.channels() * BATCH_SIZE];
		uchar batch_img_data[240*240*8 * BATCH_SIZE];
		uchar data[240*240*8];
		for(int i=0; i<8; i++){
		    for(int j=0; j<240; j++){
			    for(int k=0; k<240; k++){
			        data[i*8+j*240+k] = data_vector[i][j][k];
			    }
		    }
		}
		static vector<vector<double>> frontier_mask(240, vector<double>(240));
		for(int i=0; i<240; i++){
		    for(int j=0; j<240; j++){
				frontier_mask[i][j] = data_vector[3][i][j];
		    }
		}
			
		// const char *img_path2 = argv[3];
		// unsigned long start_time,end_load_model_time, stop_time;
		timeval start_time,end_load_model_time,end_init_time,end_run_time,end_process_time, stop_time;
		gettimeofday(&start_time, nullptr);
		// start_time = GetTickCount();
		long startt = get_sys_time_interval();

		// Load image
		// cv::Mat img = cv::imread(img_path);
		// img = get_net_work_img(img);

		// memcpy(batch_img_data, img.data, img.cols*img.rows*img.channels());
		memcpy(batch_img_data, data, 240*240*8);
		// data -> const char*

		cout << "===== load input data done =====" << endl;

		// Load RKNN Model
		model = load_model(model_path, &model_len);
		gettimeofday(&end_load_model_time, nullptr);
		// end_load_model_time = GetTickCount();
		long end_load_model = get_sys_time_interval();
		printf("end load model time:%ldms\n",end_load_model);
		ret = rknn_init(&ctx, model, model_len, 0);
		gettimeofday(&end_init_time, nullptr);
		// end_load_model_time = GetTickCount();
		long end_init = get_sys_time_interval();
		printf("end init model time:%ldms\n",end_init);
		if(ret < 0) {
		    printf("rknn_init fail! ret=%d\n", ret);
		    return -1;
		}

		////// Get Model Input Output Info
		rknn_input_output_num io_num;
		ret = rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
		if (ret != RKNN_SUCC) {
		    printf("rknn_query fail! ret=%d\n", ret);
		    return -1;
		}
		printf("model input num: %d, output num: %d\n", io_num.n_input, io_num.n_output);

		printf("input tensors:\n");
		rknn_tensor_attr input_attrs[io_num.n_input];
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
		rknn_tensor_attr output_attrs[io_num.n_output];
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

		// Set Input Data
		rknn_input inputs[1];
		memset(inputs, 0, sizeof(inputs));
		inputs[0].index = 0;
		inputs[0].type = RKNN_TENSOR_UINT8;
		// inputs[0].size = img.cols*img.rows*img.channels() * BATCH_SIZE;
		inputs[0].size = 240*240*8 * BATCH_SIZE;
		inputs[0].fmt = RKNN_TENSOR_NHWC;
		inputs[0].buf = batch_img_data;

		ret = rknn_inputs_set(ctx, io_num.n_input, inputs);
		if(ret < 0) {
		    printf("rknn_input_set fail! ret=%d\n", ret);
		    return -1;
		}

		// Run
		printf("rknn_run\n");
		ret = rknn_run(ctx, nullptr);
		if(ret < 0) {
		    printf("rknn_run fail! ret=%d\n", ret);
		    return -1;
		}

		// Get Output
		rknn_output outputs[1];
		memset(outputs, 0, sizeof(outputs));
		outputs[0].want_float = 1;
		ret = rknn_outputs_get(ctx, 1, outputs, NULL);
		if(ret < 0) {
		    printf("rknn_outputs_get fail! ret=%d\n", ret);
		    return -1;
		}

		long stop = get_sys_time_interval();
		// stop_time = GetTickCount();
		printf("detect spend time--------:%ldms\n",stop - end_init);
		printf("end detect time:%lds\n",stop);

		vector<float> output(240*240);
		int leng = output_attrs[0].n_elems/BATCH_SIZE;
		// Post Process
		for (int i = 0; i < output_attrs[0].n_elems; i++) {

		    float val = ((float*)(outputs[0].buf))[i];
		    // printf("----->%d - %f\n", i, val);
		    output[i] = val;
		    // printf("size of ouput:%d\n", output.size());
		}

		printf("[1]:%f, [2]:%f, [3]:%f\n", output[0], output[1], output[2]);
		printf("======== Getting target done ========\n");
		printf("output size!!!! = %d", output.size());
		pro_target(output, output.size(), frontier_mask);
		// output -> output
		// Release rknn_outputs
		rknn_outputs_release(ctx, 1, outputs);

		// Release
		if(ctx >= 0) {
		    rknn_destroy(ctx);
		}
		if(model) {
		    free(model);
		}
		*/
	}

    return 0;
}
