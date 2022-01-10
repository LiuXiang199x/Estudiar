#include <RlApi/include/rl_api.h>

using namespace std;
using namespace everest;
using namespace everest::planner;

RlApi::RlApi()
{
	visited_map = std::vector<std::vector<int>>(1440, std::vector<int>(1440, 0));
}

RlApi::~RlApi()
{
	
}

uint64_t RlApi::time_tToTimestamp(const time_t &t ){
    return (((uint64_t)t) * (uint64_t)10000000) + ((uint64_t)116444736*1000000000);
}

uint64_t RlApi::get_sys_time_interval(){
    timespec  tim;
    clock_gettime(CLOCK_MONOTONIC, &tim);
    return (time_tToTimestamp( tim.tv_sec ) + tim.tv_nsec/100)/10000;
}

void RlApi::printRKNNTensor(rknn_tensor_attr *attr) {
    printf("index=%d name=%s n_dims=%d dims=[%d %d %d %d] n_elems=%d size=%d fmt=%d type=%d qnt_type=%d fl=%d zp=%d scale=%f\n", 
            attr->index, attr->name, attr->n_dims, attr->dims[3], attr->dims[2], attr->dims[1], attr->dims[0], 
            attr->n_elems, attr->size, 0, attr->type, attr->qnt_type, attr->fl, attr->zp, attr->scale);
}

unsigned char* RlApi::load_model(const char *filename, int *model_size)
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


void RlApi::release_rknn(){
	// Release rknn_outputs
	rknn_outputs_release(ctx, 1, outputs);

	// Release
	if (ctx >= 0) {
		rknn_destroy(ctx);
	}
	if (model) {
		free(model);
	}
}

int RlApi::init_rknn(){
	// Load RKNN Model
	model = load_model(model_path, &model_len);

	// end_load_model_time = GetTickCount();

	ret = rknn_init(&ctx, model, model_len, 0);

	if (ret < 0) {
		printf("rknn_init fail! ret=%d\n", ret);
		return -1;
	}

	////// Get Model Input Output Info

	ret = rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
	if (ret != RKNN_SUCC) {
		printf("rknn_query fail! ret=%d\n", ret);
		return -1;
	}
	printf("model input num: %d, output num: %d\n", io_num.n_input, io_num.n_output);


	return 0;
}

// 最大池化函数
template <typename _Tp>
_Tp* MaxPolling::poll(_Tp* matrix, int matrix_w, int matrix_h, int kernel_size, int stride, bool show) {

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
void MaxPolling::showMatrix(_Tp matrix, int matrix_w, int matrix_h) {
	for (int i = 0; i < matrix_h; i++) {
		for (int j = 0; j < matrix_w; j++) {
			std::cout << matrix[i * matrix_w + j] << " ";
		}
		std::cout << std::endl;
	}
}

// 取kernel中最大值
template <typename _Tp>
_Tp MaxPolling::getMax(_Tp* matrix, int matrix_w, int matrix_h, int kernel_size, int x, int y) {
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

void MaxPolling::testMaxPolling() {
	int matrix[36] = { 1,3,1,3,5,1,4,7,5,7,9,12,1,4,6,2,5,8,6,3,9,2,1,5,8,9,2,4,6,8,4,12,54,8,0,23 };
	poll(matrix, 6, 6, 2, 2, true);
}

bool RlApi::processTarget(const int &idx, const int &idy, int &res_idx, int &res_idy) {

	// static vector<vector<double>> map(800, vector<double>(800, 0));
	printf("======== Start processing datas =======\n");
	printf("m_max_range = %f, m_min_range = %f\n", m_max_range, m_min_range);
	int size_x = m_map.getSizeX();
	int size_y = m_map.getSizeY();
	// int size_x = m_map.size();
	// int size_y = m_map[0].size();
		
	int x_origin = 200;
	int y_origin = 200;
	double tmp_value;
	int robot__x;
	int robot__y;
	int expand_type;

	printf("Original Occupancy2DMaps::====> m_map.getCell(0,0)=%f, m_map.getCell(10,16)=%f, m_map.getCell(178,354)=%f\n", m_map.getCell(0,0), m_map.getCell(10,16), m_map.getCell(178,354));

	// global map[0] = map occupancy: -1/100-->1(unexplored space/obstacles); 0-->0(free space) --- expand with 1
	// global map[1] = explored states: 0/100-->1(free space/obstacles); -1-->0(unexplored space) --- expand with 0
	// global pose = agent status on the map: 1-->robot position; 0-->No --- expand with 0
	// frontier map: 1-->frontiers; 0-->not frontiers --- expand with 0 before FBE: 1:explored spaces, 0:unexplored spaces
	// obstacles <-- 0.5 --> free space: threshold = [m_min_range, m_max_range]

	static vector<vector<int>> map_occupancy(1440, vector<int>(1440, 1));
	static vector<vector<int>> explored_states(1440, vector<int>(1440, 0));
	static vector<vector<int>> agent_status(1440, vector<int>(1440, 0));
	static vector<vector<int>> frontier_map(240, vector<int>(240, 0));
	// static vector<vector<vector<int>>> output_maps(8, vector<vector<int>>(240, vector<int>(240)));	

	int* Ocp_pooling;
	int* Expp_pooling;
	int* Visitedmap_pooling;
	int* Agentp_pooling;

	static vector<vector<int>> Ocp_crop(240, vector<int>(240));
	static vector<vector<int>> Expp_crop(240, vector<int>(240));
	static vector<vector<int>> Visitedmap_crop(240, vector<int>(240));
	static vector<vector<int>> Agentp_crop(240, vector<int>(240));

	static int Ocmap[1440000];
	static int Expmap[1440000] = {0};
	static int Agentmap[1200 * 1200] = {0};
	// memset(Ocmap, 1, sizeof(int)*1440000);
	fill(Ocmap, Ocmap+1440000, 1);

	printf("======== Getting maps data =======\n");
	// map_occupancy / explored_states / agent_status
	if (size_x == 800 && size_y == 800) {
		printf("===== Case 800*800 +++> expand type 0 =====\n");
		for (int y = 0; y < size_y; y++) {
			for (int x = 0; x < size_x; x++) {
				tmp_value = m_map.getCell(x, y);
				// tmp_value = m_map[y][x];
				// obstacles
				if (tmp_value <= m_min_range) {
					map_occupancy[y_origin+120+ y][x_origin+120+x] = 1;
					explored_states[y_origin+120 + y][x_origin+120 + x] = 1;
					Ocmap[(y_origin + y)*1200 + x_origin + x] = 1;
					Expmap[(y_origin + y)*1200 + x_origin + x] = 1;
				}
				// free space
				if (tmp_value >= m_max_range) {
					map_occupancy[y_origin+120 + y][x_origin+120 + x] = 0;
					explored_states[y_origin+120 + y][x_origin+120 + x] = 1;
					Ocmap[(y_origin + y)*1200 + x_origin + x] = 0;
					Expmap[(y_origin + y)*1200 + x_origin + x] = 1;
				}
				// unexplored space
				if (tmp_value > m_min_range && tmp_value < m_max_range) {
					map_occupancy[y_origin+120 + y][x_origin+120 + x] = 1;
					explored_states[y_origin+120 + y][x_origin+120 + x] = 0;
					Ocmap[(y_origin + y)*1200 + x_origin + x] = 1;
					Expmap[(y_origin + y)*1200 + x_origin + x] = 0;
				}

				// double float_value = map.getCell(x, y);
			}
		}
		robot__x = x_origin + idx;
		robot__y = y_origin + idy;
		agent_status[robot__y+120][robot__x+120] = 1;
		visited_map[robot__y+120][robot__x+120] = 1;
		Agentmap[robot__y*1200 + robot__x] = 1;
		Visitedmap[robot__y*1200 + robot__x] = 1;
		expand_type = 0;
	}
	if (size_x == 800 && size_y == 1200) {
		printf("===== Case 800*1200 +++> expand type 1 =====\n");
		for (int y = 0; y < size_y; y++) {
			for (int x = 0; x < size_x; x++) {
				tmp_value = m_map.getCell(x, y);
				// tmp_value = m_map[y][x];
				// obstacles
				if (tmp_value <= m_min_range) {
					map_occupancy[y+120][x_origin+120 + x] = 1;
					explored_states[y+120][x_origin+120 + x] = 1;
					Ocmap[y*1200 + x_origin + x] = 1;
					Expmap[y*1200 + x_origin + x] = 1;
				}
				// free space
				if (tmp_value >= m_max_range) {
					map_occupancy[y+120][x_origin+120 + x] = 0;
					explored_states[y+120][x_origin+120 + x] = 1;
					Ocmap[y*1200 + x_origin + x] = 0;
					Expmap[y*1200 + x_origin + x] = 1;
				}
				// unexplored space
				if (tmp_value > m_min_range && tmp_value < m_max_range) {
					map_occupancy[y+120][x_origin+120 + x] = 1;
					explored_states[y+120][x_origin+120 + x] = 0;
					Ocmap[y*1200 + x_origin + x] = 1;
					Expmap[y*1200 + x_origin + x] = 0;
				}

				// double float_value = map.getCell(x, y);
			}
		}
		robot__x = x_origin + idx;
		robot__y = idy;
		agent_status[robot__y+120][robot__x+120] = 1;
		visited_map[robot__y+120][robot__x+120] = 1;
		Agentmap[robot__y*1200 + robot__x] = 1;
		Visitedmap[robot__y*1200 + robot__x] = 1;
		expand_type = 1;
	}
	if (size_x == 1200 && size_y == 800) {
		printf("===== Case 1200*800 +++> expand type 2 =====\n");
		for (int y = 0; y < size_y; y++) {
			for (int x = 0; x < size_x; x++) {
				tmp_value = m_map.getCell(x, y);
				// tmp_value = m_map[y][x];
				// obstacles
				if (tmp_value <= m_min_range) {
					map_occupancy[y_origin +120+ y][x+120] = 1;
					explored_states[y_origin +120+ y][x+120] = 1;
					Ocmap[(y_origin + y)*1200 + x] = 1;
					Expmap[(y_origin + y)*1200 + x] = 1;
				}
				// free space
				if (tmp_value >= m_max_range) {
					map_occupancy[y_origin + y+120][x+120] = 0;
					explored_states[y_origin + y+120][x+120] = 1;
					Ocmap[(y_origin + y)*1200 + x] = 0;
					Expmap[(y_origin + y)*1200 + x] = 1;
				}
				// unexplored space
				if (tmp_value > m_min_range && tmp_value < m_max_range) {
					map_occupancy[y_origin + y+120][x+120] = 1;
					explored_states[y_origin + y+120][x+120] = 0;
					Ocmap[(y_origin + y)*1200 + x] = 1;
					Expmap[(y_origin + y)*1200 + x] = 0;
				}

				// double float_value = map.getCell(x, y);
			}
		}
		robot__x = idx;
		robot__y = y_origin + idy;
		agent_status[robot__y+120][robot__x+120] = 1;
		visited_map[robot__y+120][robot__x+120] = 1;
		Agentmap[robot__y*1200 + robot__x] = 1;
		Visitedmap[robot__y*1200 + robot__x] = 1;
		expand_type = 2;
	}
	if (size_x == 1200 && size_y == 1200) {
		printf("===== Case 1200*1200 +++> expand type 3 =====\n");
		for (int y = 0; y < size_y; y++) {
			for (int x = 0; x < size_x; x++) {
				tmp_value = m_map.getCell(x, y);
				// tmp_value = m_map[y][x];
				// obstacles
				if (tmp_value <= m_min_range) {
					map_occupancy[x+120][y+120] = 1;
					explored_states[x+120][y+120] = 1;
					Ocmap[y*1200 + x] = 1;
					Expmap[y*1200 + x] = 1;
				}
				// free space
				if (tmp_value >= m_max_range) {
					map_occupancy[x+120][y+120] = 0;
					explored_states[x+120][y+120] = 1;
					Ocmap[y*1200 + x] = 0;
					Expmap[y*1200 + x] = 1;
				}
				// unexplored space
				if (tmp_value > m_min_range && tmp_value < m_max_range) {
					map_occupancy[x+120][y+120] = 1;
					explored_states[x+120][y+120] = 0;
					Ocmap[y*1200 + x] = 1;
					Expmap[y*1200 + x] = 0;
				}

				// double float_value = m_map.getCell(x, y);
			}
		}
		robot__x = idx;
		robot__y = idy;
		agent_status[robot__y+120][robot__x+120] = 1;
		visited_map[robot__y+120][robot__x+120] = 1;
		Agentmap[robot__y*1200 + robot__x] = 1;
		Visitedmap[robot__y*1200 + robot__x] = 1;
		expand_type = 3;
	}


	printf("======== Pooling maps & Cropping maps & Getting frontier maps =======\n");
	// timeval start_maxpoolmaps, end_maxpoolmaps, start_cropmaps, end_cropmaps;
	MaxPolling pool2d;
	// maps:(1200, 1200) ---> (240, 240)

	Ocp_pooling = pool2d.poll(Ocmap, 1200, 1200, 5, 5, false);
	Expp_pooling = pool2d.poll(Expmap, 1200, 1200, 5, 5, false);
	Agentp_pooling = pool2d.poll(Agentmap, 1200, 1200, 5, 5, false);
	Visitedmap_pooling = pool2d.poll(Visitedmap, 1200, 1200, 5, 5, false);

	Ocp_crop = crop_map(map_occupancy, robot__x+120, robot__y+120, int(1));
	Expp_crop = crop_map(explored_states, robot__x+120, robot__y+120, int(0));
	Agentp_crop = crop_map(agent_status, robot__x+120, robot__y+120, int(0));
	Visitedmap_crop = crop_map(visited_map, robot__x+120, robot__y+120, int(0));

	// using crop map
	// int* frontieeer;
	// int* frontierrr;
	// frontieeer = &Expp_crop[0][0];
	// frontierrr = &Ocp_crop[0][0];
	frontier_map = get_frontier(Expp_pooling, Ocp_pooling, 240, 240);

	int flag_end = 0;
	int cout_ocp_pooling = 0;
	int cout_exp_pooling = 0;	
	int cout_agent_pooling = 0;
	int cout_visited_pooling = 0;
	int cout_ocp_crop = 0;
	int cout_exp_crop = 0;
	int cout_agent_crop = 0;
	int cout_visited_crop = 0;

	for(int i=0; i<frontier_map.size(); i++){
		for(int j=0; j<frontier_map[0].size(); j++){
			if(Ocp_crop[i][j] == 0){
				cout_ocp_crop++;
			}
			if(Expp_crop[i][j] == 1){
				cout_exp_crop++;
			}
			if(Agentp_crop[i][j] == 1){
				cout_agent_crop++;
			}
			if(Visitedmap_crop[i][j] == 1){
				cout_visited_crop++;
			}
			if(Ocp_pooling[i*240+j] == 0){
				cout_ocp_pooling++;
			}
			if(Expp_pooling[i*240+j] == 1){
				cout_exp_pooling++;
			}
			if(Agentp_pooling[i*240+j] == 1){
				cout_agent_pooling++;
			}
			if(Visitedmap_pooling[i*240+j] == 1){
				cout_visited_pooling++;
			}
			if(frontier_map[i][j] == 1){
				flag_end++;
			}
		}
	}
	printf("======> Value of frontier_map: %d ======\n======> Value of cout_ocp_pooling: %d ======\n======> Value of cout_exp_pooling: %d ======\n======> Value of cout_agent_pooling: %d ======\n======> Value of cout_visited_pooling: %d ======\n======> Value of cout_ocp_crop: %d ======\n======> Value of cout_exp_crop: %d ======\n======> Value of cout_agent_crop: %d ======\n======> Value of cout_visited_crop: %d ======\n", flag_end, cout_ocp_pooling, cout_exp_pooling, cout_agent_pooling, cout_visited_pooling, cout_ocp_crop, cout_exp_crop, cout_agent_crop, cout_visited_crop);


	printf("========== ALL DATA PREPARED ==========\n");
	printf("========== START PREDICTION ==========\n");
	static vector<float> output_prob(240 * 240);
	// predictions(output_maps, expand_type, frontier_map, res_idx, res_idy);

	for (int j = 0; j < 240; j++) {
		for (int k = 0; k < 240; k++) {
			// change data -> bath_img_data
			batch_img_data[240*240 * 0 + j * 240 + k] = Ocp_crop[j][k];
			batch_img_data[240*240 * 1 + j * 240 + k] = Expp_crop[j][k];
			batch_img_data[240*240 * 2 + j * 240 + k] = Visitedmap_crop[j][k];
			batch_img_data[240*240 * 3 + j * 240 + k] = Agentp_crop[j][k];
			batch_img_data[240*240 * 4 + j * 240 + k] = *(Ocp_pooling + j* 240 + k);
			batch_img_data[240*240 * 5 + j * 240 + k] = *(Expp_pooling + j * 240 + k);
			batch_img_data[240*240 * 6 + j * 240 + k] = *(Visitedmap_pooling +j * 240 + k);
			batch_img_data[240*240 * 7 + j * 240 + k] = *(Agentp_pooling + j * 240 + k);
		}
	}
	printf("batch_img_data[0]=%d, batch_img_data[1]=%d, batch_img_data[240*5]=%d, batch_img_data[240*240+240*2+2]=%d, batch_img_data[240*240+564]=%d, batch_img_data[240*240*2+354]=%d, batch_img_data[240*240*4+515]=%d, batch_img_data[240*240*6+514]=%d, batch_img_data[240*240*8+261]=%d, batch_img_data[240*240*10+213]=%d, batch_img_data[240*240*12+413]=%d, batch_img_data[240*240*14+113]=%d\n", batch_img_data[0], batch_img_data[1], batch_img_data[240*5], batch_img_data[240*240+240*2+2], batch_img_data[240*240+564], batch_img_data[240*240*2+354], batch_img_data[240*240*4+515], batch_img_data[240*240*6+514], batch_img_data[240*240*8+261], batch_img_data[240*240*10+213], batch_img_data[240*240*12+413], batch_img_data[240*240*14+113]);
	printf("========== ALL DATA PREPARED ==========\n");


	/////////// init_rknn ////////
	// Load RKNN Model
	model = load_model(model_path, &model_len);

	// end_load_model_time = GetTickCount();

	ret = rknn_init(&ctx, model, model_len, 0);

	if (ret < 0) {
		printf("rknn_init fail! ret=%d\n", ret);
		return -1;
	}

	////// Get Model Input Output Info
	ret = rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
	if (ret != RKNN_SUCC) {
		printf("rknn_query fail! ret=%d\n", ret);
		return -1;
	}
	printf("model input num: %d, output num: %d\n", io_num.n_input, io_num.n_output);
	/////　init_rknn done ////////

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

	memset(outputs, 0, sizeof(outputs));
	outputs[0].want_float = 1;
	ret = rknn_outputs_get(ctx, 1, outputs, NULL);
	if (ret < 0) {
		printf("rknn_outputs_get fail! ret=%d\n", ret);
		return -1;
	}


	vector<float> output(240 * 240);
	int leng = output_attrs[0].n_elems / BATCH_SIZE;
	// Post Process
	for (int i = 0; i < output_attrs[0].n_elems; i++) {

		float val = ((float*)(outputs[0].buf))[i];
		// printf("----->%d - %f\n", i, val);
		output[i] = val;
		// printf("size of ouput:%d\n", output.size());
	}


	printf("[1]:%f, [2]:%f, [3]:%f\n", output[0], output[1], output[2]);
	printf("======== Getting target done ========\n");
	pro_target(output, expand_type, frontier_map, res_idx, res_idy);

	/*
	int flag_end = 0;

	for (int i = 0; i < 240; i++) {
		for (int j = 0; j < 240; j++) {
			if (Frontier_crop[i][j] == 1) {
				flag_end = flag_end + 1;
			}
		}
	}

	if (flag_end == 0) {
		printf("There is no frontier left on the map\n");
	}
	*/

	return true;
}


vector<vector<double>> RlApi::get_inputs(int &robotx, int &roboty, int map_x, int map_y){
	printf("====== Generate random input data ======\n");
    // srand(time(0));
    static vector<vector<double>> inputs(map_x, vector<double>(map_y));

    for(int i=0; i<map_x; i++){
		for(int j=0;j<map_y;j++){
			inputs[i][j] = rand() * 1.0 / RAND_MAX;
		}
	}

	int robot_x = (rand()%100%8)*100+rand()%100;
	int robot_y = (rand()%100%8)*100+rand()%100;
	inputs[robot_x][robot_y] = 1;
	robotx = robot_x;
	roboty = robot_y;
	printf("====>robot_x=%d, robot_y=%d<====\n", robot_x, robot_y);
	printf("====> %d * %d random maps generated !!! <====\n", map_x, map_y);

    return inputs;
}


vector<vector<int>> RlApi::crop_map(vector<vector<int>> tmp, int x, int y, int padding_num) {
	printf("======== Getting croped maps =======\n");
	// static vector<vector<double>> map(1200 + 240, vector<double>(1200 + 240, padding_num));
	static vector<vector<int>> map_output(240, vector<int>(240, padding_num));

	for (int i = 0; i < 240; i++) {
		map_output[i].assign(tmp[i+y-120].begin()+x-120, tmp[i+y-120].begin()+x+120);
	}
	return map_output;
}


vector<vector<int>> RlApi::get_frontier(int *explored_map, int *occupancy_map, int row, int column) {
	printf("======== Getting frontier maps =======\n");
	// global map[0] = map occupancy: -1/100-->1(unexplored space/obstacles); 0-->0(free space) --- expand with 1
	// global map[1] = explored states: 0/100-->1(free space/obstacles); -1-->0(unexplored space) --- expand with 0
	static vector<vector<int>> map_frontier(row, vector<int>(column, 0));
	
	for (int i = 1; i < row-1; i++) {
		for (int j = 1; j < column-1; j++) {
			int tmp = *(explored_map+(i-1)*240+j) + *(explored_map+(i+1)*240+j) +*(explored_map+i*240+j+1) +*(explored_map+i*240+j-1);

			// make sure that target or mask point not in "-1"/UnexploredPlace
			if (*(explored_map+240*i + j) == 1) {
				// make sure the point is "frontier point"
				if (*(explored_map+(i-1)*240+j) ==0  or *(explored_map+(i+1)*240+j) ==0 or *(explored_map+i*240+j+1) ==0 or *(explored_map+i*240+j-1)==0){
					// filter one pixel frontier ====> short frontier
					if(tmp!=0){
						map_frontier[i][j] = 1;
					}
				}
			}

			// make sure that frontier point is in free space
			if(map_frontier[i][j]==1 && *(occupancy_map+240*i + j)==0){
					map_frontier[i][j] = 1;
			}
			if(map_frontier[i][j]==1 && *(occupancy_map+240*i + j)==1){
					map_frontier[i][j] = 0;
			}
		}
	}
	/*
	// filter short frontier otra vez
	for (int i = 1; i < row-1; i++) {
		for (int j = 1; j < column-1; j++) {
			int tmp_ = map_frontier[i][j - 1] + map_frontier[i][j + 1] + map_frontier[i - 1][j] + map_frontier[i + 1][j];
			if(tmp_==0){
				map_frontier[i][j]=0;
			}
		}
	}*/
	
	/*
	for(int i=0; i<240;i++){
		for(int j=0; j<240;j++){
			cout << map_frontier[i][j] << "";
		}
		cout << endl;
	}*/
	return map_frontier;
}


void RlApi::pro_target(vector<float> outputs, int expand_type, vector<vector<int>> mask, int &res_idx, int &res_idy) {

	printf("==== Model prediction done... Processing target ====\n");
	printf(" ouput size is: %d, mask size is %d and %d. \n", outputs.size(), mask.size(), mask[0].size());
	for (int i = 0; i < 240; i++) {
		for (int j = 0; j < 240; j++) {
			if(mask[i][j] != 1){
				outputs[i * 240 + j] = 0;
			}
		}
	}
	printf("-=-=-=-=-=-=-=-===========\n");
	double tmp_max = outputs[0];
	int tmp_index = 0;
	for (int i = 0; i < outputs.size(); i++) {
		if (tmp_max < outputs[i]) {
			tmp_max = outputs[i];
			tmp_index = i;
		}
	}
	printf("-=-=-=-=-=-=-=-===========\n");

	vector<int> target_point(2);
	// x: rows;    y: columns
	// cout << "tmp_index = " << tmp_index << endl;
	int tar_x = tmp_index % 240 * 5;
	int tar_y = tmp_index / 240 * 5;

	printf("====== Evaluate target point in frontier mask frontier_mask[x][y]=%d =======\n", mask[tmp_index / 240][tmp_index % 240]);

	// cout << "x=" << x << " y=" << y << endl;
	// transform target to original map
	if (expand_type == 0) {
		if ((tar_x - 200) >= 0) {
			target_point[0] = tar_x - 200;
		}
		if ((tar_x - 200) < 0) {
			target_point[0] = tar_x;
		}
		if ((tar_y - 200) >= 0) {
			target_point[1] = tar_y - 200;
		}
		if ((tar_y - 200) < 0) {
			target_point[1] = tar_y;
		}

	}
	if (expand_type == 1) {
		target_point[0] = tar_x;
		if ((tar_y - 200) >= 0) {
			target_point[1] = tar_y - 200;
		}
		else {
			target_point[1] = tar_y;
		}
	}
	if (expand_type == 2) {
		if ((tar_x - 200) >= 0) {
			target_point[0] = tar_x - 200;
		}
		else {
			target_point[0] = tar_x;
		}
		target_point[1] = tar_y;
	}
	if (expand_type == 3) {
		target_point[0] = tar_x;
		target_point[1] = tar_y;
	}

	
	// if transfomed target is obstacle --> use traditional algorithm
	if (m_map.getCell(tar_x, tar_y) <= m_min_range) {
		for (int i = tar_y - 2; i < tar_y + 3; i++) {
			double tmp_;
			for (int j = tar_x - 2; j < tar_x + 3; j++) {
				// tmp_ = m_map[y][x];
				tmp_ = m_map.getCell(tar_x, tar_y);
				if (tmp_ > m_min_range) {
					target_point[0] = j;
					target_point[1] = i;
					break;
				}
			}
			if (tmp_ > m_min_range) {
				break;
			}
		}
		// use traditional way

	}
	res_idx = target_point[0];
	res_idy = target_point[1];

	printf("the target x is===> %d\n", res_idx);
	printf("the target y is===> %d\n", res_idy);

	cout << "====== get target done ======\n" << endl;
	// return target_point;
}


