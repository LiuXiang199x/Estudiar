#pragma once
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
#include <algorithm>
// #include <mrpt/slam/COccupancyGridMap2D.h>

#include "rknn_api.h"

#define BATCH_SIZE 1
#define uchar unsigned char

#define img_width 64
#define img_height 64
#define img_channels 3


rknn_context ctx;
int ret;
int model_len = 0;
unsigned char* model;
const char* model_path = "/userdata/model/ckpt_precompile_20_rk161.rknn";
// printf("mode path is : %s\n", model_path);
uchar batch_img_data[240 * 240 * 8 * BATCH_SIZE];
uchar data[240 * 240 * 8];
rknn_output outputs[1];
rknn_input_output_num io_num;

namespace everest
{
	namespace planner
	{
		class RlApi{
			public:
				bool processTarget(std::vector<std::vector<double>> m_map, const int &idx, const int &idy, int &res_idx, int &res_idy);

				// for test
				std::vector<std::vector<double>> get_inputs(int &robotx, int &roboty);

				// init & release rknn ret
				void release_rknn();
				int init_rknn();

			private:
				uint64_t time_tToTimestamp(const time_t &t);
				uint64_t get_sys_time_interval();
				void printRKNNTensor(rknn_tensor_attr *attr);
				unsigned char *load_model(const char *filename, int *model_size);
				int predictions(std::vector<std::vector<std::vector<double>>> inputs_map, int expand_type,
								std::vector<std::vector<double>> mask, int &res_idx, int &res_idy);
				std::vector<std::vector<double>> crop_map(std::vector<std::vector<double>> tmp, int x, int y, double padding_num);
				std::vector<std::vector<double>> get_frontier(std::vector<std::vector<double>> explored_map,
															  std::vector<std::vector<double>> occupancy_map,
															   int row, int column);
				std::vector<std::vector<double>> filter_frontier(std::vector<std::vector<double>> map,
																 int row, int column, int minimum_size);
				void pro_target(std::vector<float> outputs, int expand_type,
								std::vector<std::vector<double>> mask, int &res_idx, int &res_idy);


			private:
				// mrpt::slam::COccupancyGridMap2D					m_map;
				double 											m_min_range;
				double 											m_max_range;
		};

		class MaxPolling {
			public:

				// 最大池化函数
				template <typename _Tp>
				_Tp* poll(_Tp* matrix, int matrix_w, int matrix_h, int kernel_size, int stride, bool show);

				template <typename _Tp>
				void showMatrix(_Tp matrix, int matrix_w, int matrix_h);

				// 取kernel中最大值
				template <typename _Tp>
				_Tp getMax(_Tp* matrix, int matrix_w, int matrix_h, int kernel_size, int x, int y);

				void testMaxPolling();
			};
	}
}
