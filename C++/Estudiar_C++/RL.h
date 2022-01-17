#pragma once
#include"alllib.h"

using namespace std;

// 结束条件
// [-30, 15]

// mrpt::slam::COccupancyGridMap2D map;
double min_range = 0.5 - 0.117;
double max_range = 0.5 + 0.059;
bool pointstate(vector<vector<double>> tmp, int row, int column);
void get_map(int robot_x, int robot_y);
vector<vector<double>> crop_map(vector<vector<double>> tmp, int x, int y, double padding_num);
vector<vector<double>> get_frontier(vector<vector<double>> tmp, int row, int column);

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

class model_prediction {
public:
	uint64_t time_tToTimestamp(const time_t& t) {
		return (((uint64_t)t) * (uint64_t)10000000) + ((uint64_t)116444736 * 1000000000);
	}

	uint64_t get_sys_time_interval() {
		timespec  tim;
		clock_gettime(CLOCK_MONOTONIC, &tim);
		return (time_tToTimestamp(tim.tv_sec) + tim.tv_nsec / 100) / 10000;
	}

	static void printRKNNTensor(rknn_tensor_attr* attr) {
		printf("index=%d name=%s n_dims=%d dims=[%d %d %d %d] n_elems=%d size=%d fmt=%d type=%d qnt_type=%d fl=%d zp=%d scale=%f\n",
			attr->index, attr->name, attr->n_dims, attr->dims[3], attr->dims[2], attr->dims[1], attr->dims[0],
			attr->n_elems, attr->size, 0, attr->type, attr->qnt_type, attr->fl, attr->zp, attr->scale);
	}

	static unsigned char* load_model(const char* filename, int* model_size)
	{
		FILE* fp = fopen(filename, "rb");
		if (fp == nullptr) {
			printf("fopen %s fail!\n", filename);
			return NULL;
		}
		fseek(fp, 0, SEEK_END);
		int model_len = ftell(fp);
		unsigned char* model = (unsigned char*)malloc(model_len);
		fseek(fp, 0, SEEK_SET);
		if (model_len != fread(model, 1, model_len, fp)) {
			printf("fread %s fail!\n", filename);
			free(model);
			return NULL;
		}
		*model_size = model_len;
		if (fp) {
			fclose(fp);
		}
		return model;
	}

	vector<vector<vector<double>>> get_inputs() {
		vector<vector<vector<double>>> inputs(8, vector<vector<double>>(240, vector<double>(240, 0)));
		return inputs;
	}

	/*-------------------------------------------
					  Main Function
	-------------------------------------------*/
	vector<float> predictions(vector<vector<vector<double>>> inputs) {
		// const int img_width = 224;
		// const int img_height = 224;
		// const int img_channels = 3;

		rknn_context ctx;
		int ret;
		int model_len = 0;
		unsigned char* model;

		const char* model_path = "../model/ckpt.85.rknn";
		const char* img_path = argv[2];
		// vector<vector<vector<double>>> data_vector;
		// data_vector = get_inputs();
		// uchar batch_img_data[img.cols*img.rows*img.channels() * BATCH_SIZE];
		uchar batch_img_data[240 * 240 * 8 * BATCH_SIZE];
		uchar data[240 * 240 * 8];
		for (int i = 0; i < 8; i++) {
			for (int j = 0; j < 240; j++) {
				for (int k = 0; k < 240; k++) {
					data[i * 8 + j * 240 + k] = inputs[i][j][k];
				}
			}
		}

		// const char *img_path2 = argv[3];
		// unsigned long start_time,end_load_model_time, stop_time;
		timeval start_time, end_load_model_time, end_init_time, end_run_time, end_process_time, stop_time;
		gettimeofday(&start_time, nullptr);
		// start_time = GetTickCount();
		long startt = get_sys_time_interval();

		// Load image
		// cv::Mat img = cv::imread(img_path);
		// img = get_net_work_img(img);

		// memcpy(batch_img_data, img.data, img.cols*img.rows*img.channels());
		memcpy(batch_img_data, data, 240 * 240 * 8);
		// data -> const char*

		cout << "===== load input data done =====" << endl;

		// Load RKNN Model
		model = load_model(model_path, &model_len);
		gettimeofday(&end_load_model_time, nullptr);
		// end_load_model_time = GetTickCount();
		long end_load_model = get_sys_time_interval();
		printf("end load model time:%ldms\n", end_load_model);
		ret = rknn_init(&ctx, model, model_len, 0);
		gettimeofday(&end_init_time, nullptr);
		// end_load_model_time = GetTickCount();
		long end_init = get_sys_time_interval();
		printf("end init model time:%ldms\n", end_init);
		if (ret < 0) {
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
		inputs[0].size = 240 * 240 * 8 * BATCH_SIZE;
		inputs[0].fmt = RKNN_TENSOR_NHWC;
		inputs[0].buf = batch_img_data;

		ret = rknn_inputs_set(ctx, io_num.n_input, inputs);
		if (ret < 0) {
			printf("rknn_input_set fail! ret=%d\n", ret);
			return -1;
		}

		// Run
		printf("rknn_run\n");
		ret = rknn_run(ctx, nullptr);
		if (ret < 0) {
			printf("rknn_run fail! ret=%d\n", ret);
			return -1;
		}

		// Get Output
		rknn_output outputs[1];
		memset(outputs, 0, sizeof(outputs));
		outputs[0].want_float = 1;
		ret = rknn_outputs_get(ctx, 1, outputs, NULL);
		if (ret < 0) {
			printf("rknn_outputs_get fail! ret=%d\n", ret);
			return -1;
		}

		long stop = get_sys_time_interval();
		// stop_time = GetTickCount();
		printf("detect spend time--------:%ldms\n", stop - end_init);
		printf("end detect time:%lds\n", stop);

		vector<float> output(240 * 240);
		int leng = output_attrs[0].n_elems / BATCH_SIZE;
		// Post Process
		for (int i = 0; i < output_attrs[0].n_elems; i++) {

			float val = ((float*)(outputs[0].buf))[i];
			// printf("----->%d - %f\n", i, val);
			output[i] = val;
			// printf("size of ouput:%d\n", output.size());
		}

		// Release rknn_outputs
		rknn_outputs_release(ctx, 1, outputs);

		// Release
		if (ctx >= 0) {
			rknn_destroy(ctx);
		}
		if (model) {
			free(model);
		}

		return output;
	}
};
// 此处必须先赋值robot pos再调用get_datas

void get_map(int robot_x, int robot_y) {

	vector<vector<double>> map(800, vector<double>(800, 0));

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

	vector<vector<double>> map_occupancy(1200, vector<double>(1200, 1));
	vector<vector<double>> explored_states(1200, vector<double>(1200, 0));
	vector<vector<double>> agent_status(1200, vector<double>(1200, 0));
	vector<vector<double>> frontier_map(1200, vector<double>(1200, 0));

	
	cout << "======== Getting maps data =======" << endl;
	// map_occupancy / explored_states / agent_status
	if (size_x == 800 && size_y == 800) {
		for (int x = 0; x < size_x; x++) {
			for (int y = 0; y < size_y; y++) {
				tmp_value = map[x][y];
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
				tmp_value = map[x][y];
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
				tmp_value = map[x][y];
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
		robot__y = robot_y;
		agent_status[robot__x][robot__y] = 1;
		expand_type = 2;
	}
	if (size_x == 1200 && size_y == 1200) {
		for (int x = 0; x < size_x; x++) {
			for (int y = 0; y < size_y; y++) {
				tmp_value = map[x][y];
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

	cout << map_occupancy.size() << endl;
	cout << map_occupancy[0].size() << endl;

	cout << "======== Getting frontier maps =======" << endl;
	// get frontier map - trick
	/*
	frontier_map = explored_states;
	for (int row = 100; row < 1100; row++) {
		for (int column = 100; column < 1100; column++) {
			frontier_map = get_frontier(frontier_map, row, column);
		}
	}
	for (int i = 0; i < frontier_map.size(); i++) {
		for (int j = 0; j < frontier_map[0].size(); j++) {
			if (frontier_map[i][j] == -2) {
				frontier_map[i][j] = 1;
			}
			else {
				frontier_map[i][j] = 0;
			}
		}
	}
	*/
	double* Ocp_pooling;
	double* Expp_pooling;
	double* Agentp_pooling;
	double* Frontp_pooling;

	vector<vector<double>> Ocp_crop(240, vector<double>(240));
	vector<vector<double>> Expp_crop(240, vector<double>(240));
	vector<vector<double>> Agentp_crop(240, vector<double>(240));
	vector<vector<double>> Frontier_crop(240, vector<double>(240));

	double Expmap[1440000];
	double Ocmap[1440000];
	double Agentmap[1200 * 1200];
	//double Frontmap[1200 * 1200];

	for (int i = 0; i < 1200; i++) {
		for (int j = 0; j < 1200; j++) {
			Ocmap[i * 1200 + j] = map_occupancy[i][j];
			Expmap[i * 1200 + j] = explored_states[i][j];
			Agentmap[i * 1200 + j] = agent_status[i][j];
			// Frontmap[i * 1200 + j] = frontier_map[i][j];
		}
	}

	cout << "======== Getting pooling maps =======" << endl;
	MaxPolling pool2d;
	// maps:(1200, 1200) ---> (240, 240)
	Ocp_pooling = pool2d.poll(Ocmap, 1200, 1200, 5, 5, false);
	Expp_pooling = pool2d.poll(Expmap, 1200, 1200, 5, 5, false);
	Agentp_pooling = pool2d.poll(Agentmap, 1200, 1200, 5, 5, false);
	//Frontp_pooling = pool2d.poll(Frontmap, 1200, 1200, 5, 5, false);

	cout << "======== Getting croped maps =======" << endl;
	// maps:(1200, 1200) ---> (240, 240)
	Ocp_crop = crop_map(map_occupancy, robot_x, robot_y, double(1));
	Expp_crop = crop_map(explored_states, robot_x, robot_y, double(0));
	Agentp_crop = crop_map(agent_status, robot_x, robot_y, double(0));
	Frontier_crop = get_frontier(explored_states, 240, 240);

	// double output_maps[8][240][240];
	vector<vector<vector<double>>> output_maps(8, vector<vector<double>>(240, vector<double>(240)));
	for (int x = 0; x < 240; x++) {
		for (int y = 0; y < 240; y++) {
			output_maps[0][x][y] = Ocp_crop[x][y];
			output_maps[1][x][y] = Expp_crop[x][y];
			output_maps[2][x][y] = Agentp_crop[x][y];
			output_maps[3][x][y] = Frontier_crop[x][y];
			output_maps[4][x][y] = *(Ocp_pooling + x * 240 + y);
			output_maps[5][x][y] = *(Expp_pooling + x * 240 + y);
			output_maps[6][x][y] = *(Agentp_pooling + x * 240 + y);
			output_maps[7][x][y] = Frontier_crop[x][y];
		}
	}
	cout << "========== ALL DATA PREPARED ==========" << endl;
	cout << "Maps shape: " << output_maps.size() << " " << output_maps[0].size() << " " << output_maps[0][0].size() << endl;
}

vector<double> get_target(int robot_x, int robot_y) {
	get_map(robot_x, robot_y);
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

void pro_target(vector<double> outputs, int expand_type, vector<vector<double>> mask) {
	double tmp_max = outputs[0];
	int tmp_index = 0;
	for (int i = 0; i < outputs.size(); i++) {
		if (tmp_max < outputs[i]) {
			tmp_max = outputs[i];
			tmp_index = i;
		}
	}

	vector<double> target_point;
	int x = tmp_index % 240 * 5;
	int y = tmp_index / 240 * 5;

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
	}
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