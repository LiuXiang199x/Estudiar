// #include "opencv2/core/core.hpp"
// #include "opencv2/imgproc/imgproc.hpp"
// #include "opencv2/highgui/highgui.hpp"
// #include <opencv2/imgproc/types_c.h>

#include "opencv2/core/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"

#include "SCYolo3Detecor.h"
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <string> 
#include <iostream>
#include <vector>
#include<sstream>
#include <iostream>
#include <stdlib.h>
#include <string.h>
#include <dirent.h>
#include <unistd.h>
#include <sys/vfs.h>

#define yolo_net_input_size 352



// const char *img_path = "./test.jpg";
using namespace std;
bool readDirFileList(const char *directory, vector<string> &files_list, bool recursion,vector<string> &file_names)
{
    DIR *dir;
    struct dirent *ptr;
    char base[1000];
    const char *base_path = directory;

    if ((dir=opendir(base_path)) == NULL)
    {
        std::cout<<"Can not open directory .....\n"<<endl;
        return false;
    }

    while ((ptr=readdir(dir)) != NULL)
    {
        if(strcmp(ptr->d_name,".")==0 || strcmp(ptr->d_name,"..")==0)
        {
            /* Current dir OR parrent dir */
            continue;
        }
        else if(ptr->d_type == 8)
        {
            /* File */
            string base_paths(base_path);
            string file_name(ptr->d_name);
            std::string str = base_paths + "/" + file_name;
            // std::string str = format("%s/%s", base_path, ptr->d_name);
            files_list.push_back(str);
            file_names.push_back(file_name);
        }
        else if(ptr->d_type == 10)
        {
            /* Link file */
        }
        else if(ptr->d_type == 4)
        {
            /* Directory */
            if(recursion)
            {
                memset(base,'\0',sizeof(base));
                strcpy(base, base_path);
                strcat(base,"/");
                strcat(base, ptr->d_name);
                readDirFileList(base, files_list, true,file_names);
            }
        }
    }
    closedir(dir);
    return true;
}

uint64_t time_tToTimestamp(const time_t &t ){
    return (((uint64_t)t) * (uint64_t)10000000) + ((uint64_t)116444736*1000000000);
}

uint64_t get_sys_time_interval(){
    timespec  tim;
    clock_gettime(CLOCK_MONOTONIC, &tim);
    return (time_tToTimestamp( tim.tv_sec ) + tim.tv_nsec/100)/10000;
}

void test_batch_data()
{
    const char *test_data_path = "/userdata/AI/detect_data/";
    // string test_data_result_path = "/userdata/AI/detect_data_result/";
    string test_data_result_path = "/tmp/AI/detect_data_result/";
    vector<string> result_img_paths;
    vector<string> file_names;
    readDirFileList(test_data_path,result_img_paths,false,file_names);

    long right_count = 0;
    printf("begin detect data test===============>!\n");
    printf("total img size is======>%d!\n",result_img_paths.size());
    for (long i = 0; i < result_img_paths.size(); i++)
    {
        string img_path = result_img_paths[i];
        cv::Mat orig_img = cv::imread(img_path);
        long long start_detect = get_sys_time_interval();
        cv::cvtColor(orig_img, orig_img, cv::COLOR_BGR2RGB);
        long long end_cvt_time = get_sys_time_interval();
        cv::Mat img = orig_img.clone();
        long long end_clone_time = get_sys_time_interval();
        if(!orig_img.data) {
            printf("cv::imread %s fail!\n", img_path);
            continue;
        }
        if(orig_img.cols != yolo_net_input_size || orig_img.rows != yolo_net_input_size) {
            printf("resize %d %d to %d %d\n", orig_img.cols, orig_img.rows, yolo_net_input_size, yolo_net_input_size);
            cv::resize(orig_img, img, cv::Size(yolo_net_input_size, yolo_net_input_size), (0, 0), (0, 0), cv::INTER_LINEAR);
        }
        long long end_resize_time = get_sys_time_interval();
        printf("are predicting index...... %d, file_name is....%s\n",i,file_names[i].c_str());
        cv::Mat predict_frame = run_yolo_model(img,orig_img);
        long long end_detect = get_sys_time_interval();
        printf("cvt spend time ===============>%lld!\n",end_cvt_time - start_detect);
        printf("clone spend time ===============>%lld!\n",end_clone_time - end_cvt_time);
        printf("resize spend time ===============>%lld!\n",end_resize_time - end_clone_time);
         printf("detect spend time ===============>%lld!\n",end_detect - start_detect);
        if(!predict_frame.empty()){
            right_count ++;
            string result_img_path = test_data_result_path + file_names[i];
            cv::imwrite(result_img_path,predict_frame);
        }
        
    }
    printf("accary is ......%.2f\n",right_count*1.0/result_img_paths.size());
    printf("end detect data test===============>!\n");
}

void test_one_pic(const char *img_path)
{
    cv::Mat orig_img = cv::imread(img_path);
    cv::Mat img = orig_img.clone();
    if(!orig_img.data) {
        printf("cv::imread %s fail!\n", img_path);
    }
    if(orig_img.cols != yolo_net_input_size || orig_img.rows != yolo_net_input_size) {
        printf("resize %d %d to %d %d\n", orig_img.cols, orig_img.rows, yolo_net_input_size, yolo_net_input_size);
        cv::resize(orig_img, img, cv::Size(yolo_net_input_size, yolo_net_input_size), (0, 0), (0, 0), cv::INTER_LINEAR);
    }
    run_yolo_model(img,orig_img);
}

int main(int argc, char** argv)
{
    init_yolo_model();
    //测试单张
    // test_one_pic(argv[1]);
    test_batch_data();

    // while (1)
    // {
    //     cv::Mat orig_img = cv::imread(img_path);
    //     cv::Mat img = orig_img.clone();
    //     if(!orig_img.data) {
    //         printf("cv::imread %s fail!\n", img_path);
    //         return -1;
    //     }
    //     if(orig_img.cols != yolo_net_input_size || orig_img.rows != yolo_net_input_size) {
    //         printf("resize %d %d to %d %d\n", orig_img.cols, orig_img.rows, yolo_net_input_size, yolo_net_input_size);
    //         cv::resize(orig_img, img, cv::Size(yolo_net_input_size, yolo_net_input_size), (0, 0), (0, 0), cv::INTER_LINEAR);
    //     }
    //     run_yolo_model(img,orig_img);
    // }
    
    return 0;
}
