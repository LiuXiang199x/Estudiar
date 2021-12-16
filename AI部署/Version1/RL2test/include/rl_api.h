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
#include <COccupancyGridMap2D.h>

#include "rknn_api.h"

using namespace std;

//using namespace cv;
////// input robot's position /////////
void get_target(int robot_x, int robot_y);

////// get target point's x y ///////
int get_target_x();
int get_target_y();


///// ignore //////
int predictions(vector<vector<vector<double>>> inputs, int expand_type, vector<vector<double>> mask);
vector<vector<double>> crop_map(vector<vector<double>> tmp, int x, int y, double padding_num);
vector<vector<double>> get_frontier(vector<vector<double>> explored_map, vector<vector<double>> occupancy_map, int row, int column);
vector<vector<double>> filter_frontier(vector<vector<double>> map, int row, int column, int minimum_size);
void pro_target(vector<float> outputs, int expand_type, vector<vector<double>> mask);
