#include "stdio.h"
#include "vector"

using namespace std;

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