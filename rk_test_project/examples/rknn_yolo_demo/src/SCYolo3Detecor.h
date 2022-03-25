#ifndef YOLOV3_DETECTOR_H
#define YOLOV3_DETECTOR_H
// #include "opencv2/core/core.hpp"
// #include "opencv2/imgproc/imgproc.hpp"
// #include "opencv2/highgui/highgui.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"

int init_yolo_model();
cv::Mat run_yolo_model(cv::Mat img, cv::Mat orig_img);

#endif