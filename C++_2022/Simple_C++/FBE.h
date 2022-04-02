#pragma once
#include"alllib.h"

bool pointstate(vector<vector<double>> tmp, int row, int column);
vector<vector<double>> crop_map(vector<vector<double>> tmp, int x, int y, double padding_num);

vector<vector<double>> get_frontier(vector<vector<double>> tmp, int row, int column) {
	// vector<vector<double>> map_frontier(240, vector<double>(240));
	for (int i = row - 1; i < row + 2; i++) {
		for (int j = column - 1; j < column + 2; j++) {
			if (tmp[i][j] == 1 && pointstate(tmp, i, j)) {
				tmp[i][j] = -2;
				tmp = get_frontier(tmp, i, j);
			}
		}
	}
	return tmp;
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

bool pointstate(vector<vector<double>> tmp, int row, int column) {
	if (tmp[row - 1][column] == 0) {
		return true;
	}
	if (tmp[row - 1][column - 1] == 0) {
		return true;
	}
	if (tmp[row - 1][column + 1] == 0) {
		return true;
	}
	if (tmp[row + 1][column] == 0) {
		return true;
	}
	if (tmp[row + 1][column + 1] == 0) {
		return true;
	}
	if (tmp[row + 1][column - 1] == 0) {
		return true;
	}
	if (tmp[row][column + 1] == 0) {
		return true;
	}
	if (tmp[row][column - 1] == 0) {
		return true;
	}
	return false;
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

void test_crop_fbe_maxpool() {

	vector<vector<double>> inputs(10, vector<double>(10));
	vector<vector<double>> outputs(10, vector<double>(10));
	double tmp[100] = { 0,0,0,0,0,0,0,0,0,0,
						0,0,0,0,0,0,0,0,0,0,
						0,0,1,0,1,1,0,1,0,0,
						0,0,1,0,0,0,0,0,0,0,
						0,0,0,0,0,0,0,0,0,0,
						0,0,0,0,1,1,1,0,0,0,
						0,0,1,0,1,1,1,0,0,0,
						0,0,0,0,1,1,1,0,0,0,
						0,0,0,0,0,0,0,0,0,0,
						0,0,0,0,0,0,0,0,0,0 };

	for (int i = 0; i < 10; i++) {
		for (int j = 0; j < 10; j++) {
			inputs[i][j] = tmp[i * 10 + j];
		}
	}

	for (int i = 0; i < 10; i++) {
		for (int j = 0; j < 10; j++) {
			cout << inputs[i][j] << " ";
		}
		cout << endl;
	}

	// Test pooling:
	MaxPolling pool;
	cout << "==================== Maxpooling test ====================" << endl;
	cout << "After pooling:" << endl;
	pool.poll(tmp, 10, 10, 2, 2, true);
	cout << endl;

	vector<vector<double>> crop_maps(1200, vector<double>(1200, 0));
	vector<vector<double>> crop_output;

	cout << "==================== Crop map test ====================" << endl;
	cout << "Before crop: " << crop_maps.size() << " " << crop_maps[0].size() << endl;
	int x = 400;
	int y = 400;
	cout << "robot_x = 400, robot_y = 400;  After crop:" << endl;
	crop_output = crop_map(crop_maps, x, y, 0);
	cout << "After crop: " << crop_output.size() << " " << crop_output[0].size() << endl;

	outputs = inputs;
	for (int row = 2; row < 8; row++) {
		for (int column = 2; column < 8; column++) {
			outputs = get_frontier(outputs, row, column);
			for (int i = 0; i < 10; i++) {
				for (int j = 0; j < 10; j++) {
					cout << outputs[i][j] << " ";
				}
				cout << endl;
			}
			cout << "=======================" << endl;
		}
	}
	for (int i = 0; i < outputs.size(); i++) {
		for (int j = 0; j < outputs[0].size(); j++) {
			if (outputs[i][j] == -2) {
				outputs[i][j] = 1;
			}
			else {
				outputs[i][j] = 0;
			}
		}
	}
	for (int i = 0; i < 10; i++) {
		for (int j = 0; j < 10; j++) {
			cout << outputs[i][j] << " ";
		}
		cout << endl;
	}
	cout << "=======================" << endl;
	//	outputs = get_frontier(inputs, 8, 8);

}
