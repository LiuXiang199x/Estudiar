#include <iostream>
#include <vector>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <fstream>
#include <iostream>
// #include <sys/time.h>
#include <string.h>
#include <uchar.h>

// #include "rknn_api.h"

using namespace std;

#define BATCH_SIZE 1
#define uchar unsigned char

// mrpt::slam::COccupancyGridMap2D map;
// [-30, 15]

double min_range = 0.5 - 0.117;
double max_range = 0.5 + 0.059;
int target_x;
int target_y;

void get_target(int robot_x, int robot_y);
vector<vector<double>> crop_map(vector<vector<double>> tmp, int x, int y, double padding_num);
vector<vector<double>> get_frontier(vector<vector<double>> explored_map, vector<vector<double>> occupancy_map, int row, int column);
vector<vector<double>> filter_frontier(vector<vector<double>> map, int row, int column, int minimum_size);
vector<int> pro_target(vector<float> outputs, int expand_type, vector<vector<double>> mask);
int get_target_x();
int get_target_y();

int get_target_x() {
	return target_x;
}

int get_target_y() {
	return target_y;
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
		int max_value = matrix[y * matrix_w + x];
		for (int i = 0; i < kernel_size; i++) {
			for (int j = 0; j < kernel_size; j++) {
				if (max_value < matrix[matrix_w * (y + i) + x + j]) {
					max_value = matrix[matrix_w * (y + i) + x + j];
				}
			}
		}
		return max_value;
	}

	void testMaxPolling() {
		int matrix[36] = { 1,3,1,3,5,1,4,7,5,7,9,12,1,4,6,2,5,8,6,3,9,2,1,5,8,9,2,4,6,8,4,12,54,8,0,23 };
		poll(matrix, 6, 6, 2, 2, true);
	}
};

void get_target(int robot_x, int robot_y) {

	static vector<vector<double>> map(800, vector<double>(800, 0));
	printf("we are in func get_target");
	int size_x = 800;
	int size_y = 800;
	// int size_x = map.getSizeX();
	// int size_y = map.getSizeY();
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


	printf("======== Getting maps data =======");
	// map_occupancy / explored_states / agent_status
	if (size_x == 800 && size_y == 800) {
		for (int x = 0; x < size_x; x++) {
			for (int y = 0; y < size_y; y++) {
				// tmp_value = map[x][y];
				tmp_value = 0.4;
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
		robot__x = x_origin + robot_x;
		robot__y = y_origin + robot_y;
		agent_status[robot__x][robot__y] = 1;
		expand_type = 0;
	}
	if (size_x == 800 && size_y == 1200) {
		for (int x = 0; x < size_x; x++) {
			for (int y = 0; y < size_y; y++) {
				// tmp_value = map[x][y];
				tmp_value = 0.4;
				// obstacles
				if (tmp_value <= min_range) {
					map_occupancy[x_origin + x][y] = 1;
					explored_states[x_origin + x][y] = 1;
				}
				// free space
				if (tmp_value >= max_range) {
					map_occupancy[x_origin + x][y] = 0;
					explored_states[x_origin + x][y] = 1;
				}
				// unexplored space
				if (tmp_value > min_range && tmp_value < max_range) {
					map_occupancy[x_origin + x][y] = 1;
					explored_states[x_origin + x][y] = 0;
				}

				// double float_value = map.getCell(x, y);
			}
		}
		robot__x = x_origin + robot_x;
		robot__y = robot_y;
		agent_status[robot__x][robot__y] = 1;
		expand_type = 1;
	}
	if (size_x == 1200 && size_y == 800) {
		for (int x = 0; x < size_x; x++) {
			for (int y = 0; y < size_y; y++) {
				// tmp_value = map[x][y];
				tmp_value = 0.4;
				// obstacles
				if (tmp_value <= min_range) {
					map_occupancy[x][y_origin + y] = 1;
					explored_states[x][y_origin + y] = 1;
				}
				// free space
				if (tmp_value >= max_range) {
					map_occupancy[x][y_origin + y] = 0;
					explored_states[x][y_origin + y] = 1;
				}
				// unexplored space
				if (tmp_value > min_range && tmp_value < max_range) {
					map_occupancy[x][y_origin + y] = 1;
					explored_states[x][y_origin + y] = 0;
				}

				// double float_value = map.getCell(x, y);
			}
		}
		robot__x = robot_x;
		robot__y = y_origin + robot_y;
		agent_status[robot__x][robot__y] = 1;
		expand_type = 2;
	}
	if (size_x == 1200 && size_y == 1200) {
		for (int x = 0; x < size_x; x++) {
			for (int y = 0; y < size_y; y++) {
				// tmp_value = map[x][y];
				tmp_value = 0.4;
				// obstacles
				if (tmp_value <= min_range) {
					map_occupancy[x][y] = 1;
					explored_states[x][y] = 1;
				}
				// free space
				if (tmp_value >= max_range) {
					map_occupancy[x][y] = 0;
					explored_states[x][y] = 1;
				}
				// unexplored space
				if (tmp_value > min_range && tmp_value < max_range) {
					map_occupancy[x][y] = 1;
					explored_states[x][y] = 0;
				}

				// double float_value = map.getCell(x, y);
			}
		}
		robot__x = robot_x;
		robot__y = robot_y;
		agent_status[robot__x][robot__y] = 1;
		expand_type = 3;
	}

	printf("======== Getting frontier maps =======");
	frontier_map = get_frontier(explored_states, map_occupancy, 1200, 1200);

	printf("======== Pooling & Cropping maps =======");
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

	for (int i = 0; i < 1200; i++) {
		for (int j = 0; j < 1200; j++) {
			Ocmap[i * 1200 + j] = map_occupancy[i][j];
			Expmap[i * 1200 + j] = explored_states[i][j];
			Agentmap[i * 1200 + j] = agent_status[i][j];
			Frontmap[i * 1200 + j] = frontier_map[i][j];
		}
	}

	printf("======== Getting pooling maps =======");
	MaxPolling pool2d;
	// maps:(1200, 1200) ---> (240, 240)
	Ocp_pooling = pool2d.poll(Ocmap, 1200, 1200, 5, 5, false);
	Expp_pooling = pool2d.poll(Expmap, 1200, 1200, 5, 5, false);
	Agentp_pooling = pool2d.poll(Agentmap, 1200, 1200, 5, 5, false);
	Frontp_pooling = pool2d.poll(Frontmap, 1200, 1200, 5, 5, false);

	static vector<vector<double>> front_map_pool(240, vector<double>(240));
	for (int i = 0; i < 240; i++) {
		for (int j = 0; j < 240; j++) {
			front_map_pool[i][j] = Frontp_pooling[i * 240 + j];
		}
	}
	front_map_pool = filter_frontier(front_map_pool, 240, 240, 2);

	printf("======== Getting croped maps =======");
	// maps:(1200, 1200) ---> (240, 240)
	Ocp_crop = crop_map(map_occupancy, robot_x, robot_y, double(1));
	Expp_crop = crop_map(explored_states, robot_x, robot_y, double(0));
	Agentp_crop = crop_map(agent_status, robot_x, robot_y, double(0));
	Frontier_crop = crop_map(frontier_map, robot_x, robot_y, double(0));
	Frontier_crop = filter_frontier(Frontier_crop, 240, 240, 2);

	// double output_maps[8][240][240];
	static vector<vector<vector<double>>> output_maps(8, vector<vector<double>>(240, vector<double>(240)));
	for (int x = 0; x < 240; x++) {
		for (int y = 0; y < 240; y++) {
			output_maps[0][x][y] = Ocp_crop[x][y];
			output_maps[1][x][y] = Expp_crop[x][y];
			output_maps[2][x][y] = Agentp_crop[x][y];
			output_maps[3][x][y] = Frontier_crop[x][y];
			output_maps[4][x][y] = *(Ocp_pooling + x * 240 + y);
			output_maps[5][x][y] = *(Expp_pooling + x * 240 + y);
			output_maps[6][x][y] = *(Agentp_pooling + x * 240 + y);
			output_maps[7][x][y] = front_map_pool[x][y];
		}
	}
	printf("========== ALL DATA PREPARED ==========");

	int flag_end = 0;

	for (int i = 0; i < 240; i++) {
		for (int j = 0; j < 240; j++) {
			if (Frontier_crop[i][j] == 1) {
				flag_end = flag_end + 1;
			}
		}
	}

	if (flag_end == 0) {
		printf("There is no frontier left on the map");
	}

	printf("========== START PREDICTION ==========");
	vector<float> output_prob(240 * 240);
	// predictions(output_maps, expand_type, Frontier_crop);
}

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
			map_output[i][j] = map[robot_y - 120 + i][robot_x - 120 + j];
		}
	}
	return map_output;
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

vector<vector<double>> filter_frontier(vector<vector<double>> map, int row, int column, int minimum_size) {
	for (int i = minimum_size; i < row - minimum_size; i++) {
		for (int j = minimum_size; j < column - minimum_size; j++) {
			if (map[i][j] == 1) {
				double tmp = 0;
				for (int m = i - minimum_size; m < i + minimum_size + 1; m++) {
					for (int n = j - minimum_size; n < j + minimum_size + 1; n++) {
						tmp = tmp + map[m][n];
					}
				}
				if (tmp <= 2) {
					map[i][j] = 0;
				}
			}
		}
	}
	return map;
}

vector<int> pro_target(vector<float> outputs, int expand_type, vector<vector<double>> mask) {
	for (int i = 0; i < 240; i++) {
		for (int j = 0; j < 240; j++) {
			outputs[i * 240 + j] = outputs[i * 240 + j] * mask[i][j];
		}
	}

	double tmp_max = outputs[0];
	int tmp_index = 0;
	for (int i = 0; i < outputs.size(); i++) {
		if (tmp_max < outputs[i]) {
			tmp_max = outputs[i];
			tmp_index = i;
		}
	}

	vector<int> target_point;
	// x: rows;    y: columns
	int y = tmp_index % 240 * 5;
	int x = tmp_index / 240 * 5;

	// transform target to original map
	if (expand_type == 0) {
		if ((x - 200) >= 0) {
			target_point[0] = x - 200;
		}
		if ((x - 200) < 0) {
			target_point[0] = x;
		}
		if ((y - 200) >= 0) {
			target_point[1] = y - 200;
		}
		if ((y - 200) < 0) {
			target_point[1] = y;
		}
	}
	if (expand_type == 1) {
		target_point[0] = x;
		if ((y - 200) >= 0) {
			target_point[1] = y - 200;
		}
		else {
			target_point[1] = y;
		}
	}
	if (expand_type == 2) {
		if ((x - 200) >= 0) {
			target_point[0] = x - 200;
		}
		else {
			target_point[0] = x;
		}
		target_point[1] = y;
	}
	if (expand_type == 3) {
		target_point[0] = x;
		target_point[1] = y;
	}
	/*
	// if transfomed target is obstacle --> use traditional algorithm
	if (map[x][y] <= min_range) {
		for (int i = x - 2; i < x + 3; i++) {
			double tmp_;
			for (int j = y - 2; j < y + 3; j++) {
				tmp_ = map[i][j];
				if (tmp_ > min_range) {
					target_point[0] = i;
					target_point[1] = j;
					break;
				}
			}
			if (tmp_ > min_range) {
				break;
			}
		}
		// use traditional way

	}
	*/
	return target_point;
}

int main() {
	printf("start to test!!");
	get_target(241, 445);
	return 0;
}
