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
	frontier_map = get_frontier(explored_states, map_occupancy, 1200, 1200);

	
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

vector<vector<double>> get_frontier(vector<vector<double>> explored_map,
	vector<vector<double>> occupancy_map, int row, int column) {

	// global map[0] = map occupancy: -1/100-->1(unexplored space/obstacles); 0-->0(free space) --- expand with 1
	// global map[1] = explored states: 0/100-->1(free space/obstacles); -1-->0(unexplored space) --- expand with 0
	vector<vector<double>> map_frontier(1200, vector<double>(1200, 0));

	for (int i = 1; i < 1199; i++) {
		for (int j = 1; j < 1199; j++) {
			double tmp = explored_map[i][j - 1] + explored_map[i][j + 1] + explored_map[i - 1][j] + explored_map[i + 1][j];
			if (explored_map[i][j] == 0 && tmp != 0) {
				map_frontier[i][j] = 1;
			}
		}
	}

	for (int i = 0; i < 1200; i++) {
		for (int j = 0; j < 1200; j++) {
			map_frontier[i][j] = map_frontier[i][j] * occupancy_map[i][j];
		}
	}

	return map_frontier;
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
