/*
            AI_ROOM                = 2000, // room flag
            AI_ROOM_BEDROOM        = 2001,    //////
            AI_ROOM_RESTAURANT     = 2002,    
            AI_ROOM_TOILET         = 2003,    //////
            AI_ROOM_CORRIDOR       = 2004,
            AI_ROOM_KITCHEN        = 2005,    //////
            AI_ROOM_LIVING_ROOM    = 2006,    //////
            AI_ROOM_BALCONY        = 2007,
            AI_ROOM_OTHERS         = 2020,    //////
*/


#include "SceneNet.h"
#include "aa.h"

using namespace std;
using namespace everest;
using namespace everest::ai;
using namespace Eigen;
using namespace cv;


#define BATCH_SIZE 1
#define uchar unsigned char


SceneNet::SceneNet(){

    printf(" ====== Start rknn model ======\n");
    init_rknn();
    printf(" ====== Load rknn model done ======\n");

}
SceneNet::~SceneNet(){
    // Release
    printf("====== free model ======\n");
    if(m_ctx >= 0) {
        //rknn_destroy(m_ctx);
    }
    if(m_model) {
        free(m_model);
    }
}

int SceneNet::init_rknn()
{
    // const int img_width = 416;
    // const int img_height = 416;
    // const int img_channels = 3;

    int ret=0;
    int model_len = 0;

    // Load RKNN Model
    printf("Loading model ...\n");
    m_model = load_model(model_path, &model_len);
    ret = rknn_init(&m_ctx, m_model, model_len, 0);
    if(ret < 0) {
        printf("rknn_init fail! ret=%d\n", ret);
        return -1;
    }
    return 0;
}

/*-------------------------------------------
                  Functions
-------------------------------------------*/


// 	double tmp_Nano[13] = {1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0};
// 	double tmp_Scene[512] = {1};
int SceneNet::sceneClass(float *Nanodet_res, float *Scene_res){

    Matrix<float, 512, 1> resultNano;
    Matrix<float, 1024, 1> resultConcat;
    Matrix<float, 13, 1> Nanodata;
    Matrix<float, 5, 1> result_predict;

    printf(" process nanpdet_res and scene_res......\n");
    for(int i=0; i<13; i++){
        Nanodata(i) = *(Nanodet_res + i);
        // cout << *(Scene_res+i) << endl;
    }

    resultNano = obj_fc_weight * Nanodata + obj_fc_bias;  // 512 features

    for(int i=0; i<1024; i++){
        if(i < 512){
            resultConcat(i) = resultNano(i);
        }
        else{
            resultConcat(i) = *(Scene_res+i-512);
        }
    }

    cout << resultConcat.size() << endl;
    // resultConcat = memcpy
    result_predict = class_fc_weight * resultConcat + class_fc_bias;
    cout << result_predict.size() << endl;
    // cout << result_predict << endl;
    
    float total_prob = 0;
    float model_output[5];
    float max_index = 0;
    for(int i=0; i<result_predict.size(); i++){
        // cout << result_predict(i) << endl;
        total_prob = total_prob + exp(result_predict(i));
    }

    // cout << "total_prob: " << total_prob << endl;

    for(int i=0; i<result_predict.size(); i++){
        model_output[i] = exp(result_predict(i)) / total_prob;
        if(i>0){
            if(model_output[i]>model_output[i-1]){
                max_index = i;
            }
            else{
                model_output[i] = model_output[i-1];
            }
        }
    }

    // ['bed_room', 'dining_room', 'drawing_room', 'others', 'toilet_room']
    cout << "max index: " << max_index << endl;
    if(max_index==0){
        roomType = AI_ROOM_BEDROOM;
    }
    else if(max_index==1){
        roomType = AI_ROOM_RESTAURANT;
    }
    else if(max_index==2){
        roomType = AI_ROOM_LIVING_ROOM;
    }
    else if(max_index==3){
        roomType = AI_ROOM_OTHERS;
    }
    else if(max_index==4){
        roomType = AI_ROOM_TOILET;
    }

    cout << "==============> RoomType: " << roomType << endl;
    
    return roomType;

}

int64_t getCurrentLocalTimeStamp()
{
    std::chrono::time_point<std::chrono::system_clock, std::chrono::milliseconds> tp = std::chrono::time_point_cast<std::chrono::milliseconds>(std::chrono::system_clock::now());
    auto tmp = std::chrono::duration_cast<std::chrono::milliseconds>(tp.time_since_epoch());
    return tmp.count();

    // return std::chrono::duration_cast(std::chrono::system_clock::now().time_since_epoch()).count();
};

uint64_t SceneNet::time_tToTimestamp(const time_t &t ){
    return (((uint64_t)t) * (uint64_t)10000000) + ((uint64_t)116444736*1000000000);
}

uint64_t SceneNet::get_sys_time_interval(){
    timespec  tim;
    clock_gettime(CLOCK_MONOTONIC, &tim);
    return (time_tToTimestamp( tim.tv_sec ) + tim.tv_nsec/100)/10000;
}

void SceneNet::printRKNNTensor(rknn_tensor_attr *attr) {
    printf("index=%d name=%s n_dims=%d dims=[%d %d %d %d] n_elems=%d size=%d fmt=%d type=%d qnt_type=%d fl=%d zp=%d scale=%f\n", 
            attr->index, attr->name, attr->n_dims, attr->dims[3], attr->dims[2], attr->dims[1], attr->dims[0], 
            attr->n_elems, attr->size, 0, attr->type, attr->qnt_type, attr->fl, attr->zp, attr->scale);
}

unsigned char *SceneNet::load_model(const char *filename, int *model_size)
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


int SceneNet::runSceneNet(cv::Mat& src, float *Nanodet_res)
{
    // INITIALIZE ROOM TYPE
    roomType = AI_ROOM;

    timeval start_time,end_load_model_time,end_init_time,end_run_time,end_process_time, stop_time;
    gettimeofday(&start_time, nullptr);
    // start_time = GetTickCount();

    // calc start time
    long startt = get_sys_time_interval();

    cv::Mat img = src;
    // long startt2 = get_sys_time_interval();

    // cout << "read_img: " << startt2 - startt << endl;
    int ret=0;
    int model_len = 0;

    if(!img.data) {
        printf("cv::imread fail!\n");
        return -1;
    }

    // Get Model Input Output Info
    rknn_input_output_num io_num;
    ret = rknn_query(m_ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
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
        ret = rknn_query(m_ctx, RKNN_QUERY_INPUT_ATTR, &(input_attrs[i]), sizeof(rknn_tensor_attr));
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
        ret = rknn_query(m_ctx, RKNN_QUERY_OUTPUT_ATTR, &(output_attrs[i]), sizeof(rknn_tensor_attr));
        if (ret != RKNN_SUCC) {
            printf("rknn_query fail! ret=%d\n", ret);
            return -1;
        }
        printRKNNTensor(&(output_attrs[i]));
    }

    uchar batch_img_data[img.cols*img.rows*img.channels() * BATCH_SIZE];
    memcpy(batch_img_data, img.data, img.cols*img.rows*img.channels());

    // Set Input Data
    rknn_input inputs[1];
    memset(inputs, 0, sizeof(inputs));
    inputs[0].index = 0;
    inputs[0].type = RKNN_TENSOR_UINT8;
    inputs[0].size = img.cols*img.rows*img.channels();
    inputs[0].fmt = RKNN_TENSOR_NHWC;
    inputs[0].buf = img.data;

    printf("size of inputs = %d\n", sizeof(inputs));
    ret = rknn_inputs_set(m_ctx, io_num.n_input, inputs);
    if(ret < 0) {
        printf("rknn_input_set fail! ret=%d\n", ret);
        return -1;
    }

    // Run
    printf("rknn_run\n");
    ret = rknn_run(m_ctx, nullptr);
    if(ret < 0) {
        printf("rknn_run fail! ret=%d\n", ret);
        return -1;
    }

    // Get Output
    rknn_output outputs[1];
    memset(outputs, 0, sizeof(outputs));
    outputs[0].want_float = 1;
 //   outputs[1].want_float = 1;
    ret = rknn_outputs_get(m_ctx, io_num.n_output, outputs, NULL);
    if(ret < 0) {
        printf("rknn_outputs_get fail! ret=%d\n", ret);
        return -1;
    }

    int leng = output_attrs[0].n_elems / BATCH_SIZE;
    float model_output[512];
	// Post Process
	for (int i = 0; i < output_attrs[0].n_elems; i++) {

		// float val = ((float*)(outputs[0].buf))[i];
		// printf("----->%d - %f\n", i, val);
		model_output[i] = ((float*)(outputs[0].buf))[i];
		// printf("size of ouput:%d\n", output.size());
	}

    // Release rknn_outputs
    rknn_outputs_release(m_ctx, 1, outputs);
    printf("====== extraction features done | start classification ======\n");
    int room_type = 0;
    room_type = sceneClass(Nanodet_res, model_output);

    // Endding time
    long endd = get_sys_time_interval();
    cout << "===========> All the time(read img + resnet18): " << endd - startt << endl;

    return room_type;
}
