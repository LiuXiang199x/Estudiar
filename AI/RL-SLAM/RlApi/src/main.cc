#include <RlApi/include/rl_api.h>

int main(){
	everest::planner::RlApi test;
	int res_x, res_y;
	test.processTarget(400, 400, res_x, res_y);
	return 0;
}
