#include"generate_inputs.h"
#include"math.h"
using namespace std;

int main() {
	srand((unsigned)time(NULL));//time()用系统时间初始化种。为rand()生成不同的随机种子。
	// cout <<(time(0)) << endl;
	// test();
	for (int i = 0; i < 5; i++) {
		int RanddomNumber;
		
		RanddomNumber = rand() % 100 + 1;//生成1~100随机数

		cout << RanddomNumber << endl;
	}

	return 0;
}