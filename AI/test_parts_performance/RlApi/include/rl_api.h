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
// #include <COccupancyGridMap2D.h>
// #include <mrpt/slam/COccupancyGridMap2D.h>

#include "rknn_api.h"

#define BATCH_SIZE 1
#define uchar unsigned char

#define img_width 64
#define img_height 64
#define img_channels 3

namespace everest
{
	namespace planner
	{
		class RlApi{
			public:
				bool processTarget(const int &idx, const int &idy, int &res_idx, int &res_idy);
			private:
				uint64_t time_tToTimestamp(const time_t &t);
				uint64_t get_sys_time_interval();
				void printRKNNTensor(rknn_tensor_attr *attr);
				unsigned char *load_model(const char *filename, int *model_size);
				std::vector<std::vector<std::vector<double>>> get_inputs();
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
