#include <rl_api.h>

using namespace std;

int main(){
	everest::planner::RlApi test;
	int robotxxx = 0;
	int robotyyy = 0;
	int result_x;
	int result_y;
	vector<vector<double>> inputss;

	test.init_rknn();

	for(int i=0; i<10; i++){
		inputss = test.get_inputs(robotxxx, robotyyy);

		test.processTarget(inputss, robotxxx, robotyyy, result_x, result_y);
		// test.processTarget(400, 400, res_x, res_y);

	}
	test.release_rknn();
	return 0;
}
