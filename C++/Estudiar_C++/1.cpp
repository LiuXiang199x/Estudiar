#include"generate_inputs.h"
#include"math.h"
using namespace std;

int main() {
	srand((unsigned)time(NULL));//time()��ϵͳʱ���ʼ���֡�Ϊrand()���ɲ�ͬ��������ӡ�
	// cout <<(time(0)) << endl;
	// test();
	for (int i = 0; i < 5; i++) {
		int RanddomNumber;
		
		RanddomNumber = rand() % 100 + 1;//����1~100�����

		cout << RanddomNumber << endl;
	}

	return 0;
}