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

vector<int> CutArrs(vector<int>& Arrs, int begin, int end){ // begin <= end;
        //if(end > Arrs.size()) return;
        vector<int> result;
        result.assign(Arrs.begin() + begin, Arrs.begin() + end);
        return result;
}

/*
先直接一波bug，不管他是否在边缘区域了
// tmp[map_w*y + x]
vector<vector<double>> crop_map(vector<vector<double>> tmp, int x, int y, double padding_num) {
	vector<vector<double>> map(1200 + 240, vector<double>(1200 + 240, padding_num));
	vector<vector<double>> map_output(240, vector<double>(240));
	int robot_x = x + 120;
	int robot_y = y + 120;

	for (int i = 0; i < 1200; i++) {
		map[120 + i].assign(tmp[i].begin(), tmp[i].end());
	}

	for (int i = 0; i < 240; i++) {
		for (int j = 0; j < 240; j++) {
			map_output[i][j] = map[robot_y - 120 + i][robot_x - 120 + j];
		}
	}
	return map_output;
}
*/

vector<vector<double>> crop_map(svector<vector<double>> tmp, int x, int y, double padding_num) {
	static vector<vector<double>> map(1200 + 240, vector<double>(1200 + 240, padding_num));
	static vector<vector<double>> map_output(240, vector<double>(240));
	int robot_x = x + 120;
	int robot_y = y + 120;


	for (int i = 0; i < 240; i++) {
		map_output[i].assign(tmp[i+robot_x-120].begin()+robot_y-120, tmp[i].begin()+robot_y+120);
		
	}
	return map_output;
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
	// visited map
	// global pose = agent status on the map: 1-->robot position; 0-->No --- expand with 0
	

	// frontier map: 1-->frontiers; 0-->not frontiers --- expand with 0 before FBE: 1:explored spaces, 0:unexplored spaces
	// obstacles <-- 0.5 --> free space: threshold = [min_range, max_range]

	static vector<vector<double>> map_occupancy(1200, vector<double>(1200, 1));
	static vector<vector<double>> explored_states(1200, vector<double>(1200, 0));
	static vector<vector<double>> agent_status(1200, vector<double>(1200, 0));
	static vector<vector<double>> frontier_map(1200, vector<double>(1200, 0));
	
	printf("======== Getting maps data =======\n");
	// long startt_getoriginmap = get_sys_time_interval();
	// map_occupancy / explored_states / agent_status
	if (size_x == 800 && size_y == 800) {
		for (int x = 0; x < size_x; x++) {
			for (int y = 0; y < size_y; y++) {
				tmp_value = map_data[x][y];
				/// tmp_value = 0.4;
				// obstacles
				if (tmp_value <= min_range) {
					map_occupancy[x_origin + x][y_origin + y] = 1;
					explored_states[x_origin + x][y_origin + y] = 1;
				}
				// free space
				if (tmp_value >= max_range) {
					map_occupancy[x_origin + x][y_origin + y] = 0;
					explored_states[x_origin + x][y_origin + y] = 1;
				}
				// unexplored space
				if (tmp_value > min_range && tmp_value < max_range) {
					map_occupancy[x_origin + x][y_origin + y] = 1;
					explored_states[x_origin + x][y_origin + y] = 0;
				}

				// double float_value = map.getCell(x, y);
			}
		}
		robot__x = x_origin + idx;
		robot__y = y_origin + idy;
		agent_status[robot__x][robot__y] = 1;
		expand_type = 0;
	}
	// long end_getoriginmap = get_sys_time_interval();
	// printf("Getting original map datas ===================> :%ldms\n", end_getoriginmap - startt_getoriginmap);
	printf("\n");


	printf("======== Pooling & Cropping maps =======\n");
	// timeval start_maxpoolmaps, end_maxpoolmaps, start_cropmaps, end_cropmaps;
	double* Ocp_pooling;
	double* Expp_pooling;
	double* Agentp_pooling;
	double* Frontp_pooling;

	static vector<vector<double>> Ocp_crop(240, vector<double>(240));
	static vector<vector<double>> Expp_crop(240, vector<double>(240));
	static vector<vector<double>> Agentp_crop(240, vector<double>(240));
	static vector<vector<double>> Frontier_crop(240, vector<double>(240));

	static double Expmap[1440000];
	static double Ocmap[1440000];
	static double Agentmap[1200 * 1200];
	static double Frontmap[1200 * 1200];


	printf("======== Getting croped maps =======\n");
	// long start_cropmaps = get_sys_time_interval();
	// maps:(1200, 1200) ---> (240, 240)
	Ocp_crop = crop_map(map_occupancy, robot__x, robot__y, double(1));
	Expp_crop = crop_map(explored_states, robot__x, robot__y, double(0));
	Agentp_crop = crop_map(agent_status, robot__x, robot__y, double(0));
	Frontier_crop = crop_map(frontier_map, robot__x, robot__y, double(0));
	// Frontier_crop = filter_frontier(Frontier_crop, 240, 240, 2);
	// long end_cropmaps = get_sys_time_interval();
	// printf("Getting croped map datas ===================> :%ldms\n", end_cropmaps - start_cropmaps);
	printf("\n");

	vector<vector<vector<double>>> output_maps;
	return output_maps;
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

	}

    return 0;
}
