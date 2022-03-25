#ifndef LIQUID_H
#define LIQUID_H
//#include "commdef.h"
// matrix_cal1.cpp
#include <iostream>
#include "Eigen/Eigen"
#include <vector>
#include <math.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <fstream>
#include <iostream>
#include <sys/time.h>
#include <string.h>
#include <uchar.h>
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"

#include <aa.h>   // everest/ai.h
#include "rknn_api.h"

#include <vector>
#include <string>
#include <math.h>


namespace everest{
    namespace ai{
        class SceneNet{
            public:
                int runSceneNet(float *Nanodet_res);
            
            public:
                Eigen::Matrix<float, 5, 1> class_fc_bias;
                Eigen::Matrix<float, 512, 1> obj_fc_bias;
                Eigen::Matrix<float, 5, 1024> class_fc_weight;
                Eigen::Matrix<float, 512, 13> obj_fc_weight;
                
                TAIObjectClass roomType;
                // rknn_context m_ctx;
                rknn_context m_ctx;
                unsigned char *m_model;
                const char *model_path = "/tmp/test/rknn_testModel/model/abc.rknn";
                const char *img_path = "/tmp/test/rknn_testModel/model/test_toilet.jpg";
                
            public:
                // load params of fc layers
                void load_params();
                // init rk models
                int init_rknn();
                void printRKNNTensor(rknn_tensor_attr *attr);
                unsigned char *load_model(const char *filename, int *model_size);
                // init params
                SceneNet();
                ~SceneNet();

            public:
                // get model output 13: model_output1->upSample +(concat)+ nanoplus_output->upSample -> softmax -> five classes
                void sceneClass(float *Nanodet_res, float *Scene_res);
                uint64_t time_tToTimestamp(const time_t &t );
                uint64_t get_sys_time_interval();
        };
    }
}
#endif // LIQUID_H
