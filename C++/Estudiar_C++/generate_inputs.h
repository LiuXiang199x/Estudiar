#pragma once
#include"alllib.h"
vector<vector<vector<double>>> get_inputs() {
	// srand(time(0)); 这个要在最主要的程序里面去声明，不然每次进来运行得到的数据还是随机生成的一样的
	vector<vector<vector<double>>> inputs(8, vector<vector<double>>(10, vector<double>(10, 0)));

	for (int i = 0; i < 10; i++) {
		for (int j = 0; j < 10; j++) {
			inputs[0][i][j] = (rand() & 10001) / 10000.0;
			inputs[1][i][j] = (rand() & 10001) / 10000.0;
			inputs[3][i][j] = (rand() & 10001) / 10000.0;
			inputs[4][i][j] = (rand() & 10001) / 10000.0;
			inputs[5][i][j] = (rand() & 10001) / 10000.0;
			inputs[7][i][j] = (rand() & 10001) / 10000.0;
			if (i == 45 && j == 45) {
				inputs[2][i][j] = 1;
				inputs[6][i][j] = 1;
			}
			if (inputs[0][i][j] >= 0.5) {
				inputs[0][i][j] = 1;
			}
			if (inputs[0][i][j] < 0.5) {
				inputs[0][i][j] = 0;
			}
			if (inputs[1][i][j] >= 0.5) {
				inputs[1][i][j] = 1;
			}
			if (inputs[1][i][j] < 0.5) {
				inputs[1][i][j] = 0;
			}
			if (inputs[3][i][j] >= 0.5) {
				inputs[3][i][j] = 1;
			}
			if (inputs[3][i][j] < 0.5) {
				inputs[3][i][j] = 0;
			}
			if (inputs[4][i][j] >= 0.5) {
				inputs[4][i][j] = 1;
			}
			if (inputs[4][i][j] < 0.5) {
				inputs[4][i][j] = 0;
			}
			if (inputs[5][i][j] >= 0.5) {
				inputs[5][i][j] = 1;
			}
			if (inputs[5][i][j] < 0.5) {
				inputs[5][i][j] = 0;
			}
			if (inputs[7][i][j] >= 0.5) {
				inputs[7][i][j] = 1;
			}
			if (inputs[7][i][j] < 0.5) {
				inputs[7][i][j] = 0;
			}
			if (i == 5 && j == 5) {
				inputs[2][i][j] = 1;
				inputs[6][i][j] = 1;
			}
		}
	}


	return inputs;
}

void test() {
	// srand(time(0));
	srand((unsigned)time(NULL));//time()用系统时间初始化种。为rand()生成不同的随机种子。
	for (int n = 0; n < 3; n++) {
		vector<vector<vector<double>>> aaa(8, vector<vector<double>>(10, vector<double>(10, 0)));
		aaa = get_inputs();
		for (int i = 0; i < 8; i++) {
			for (int j = 0; j < 10; j++) {
				for (int k = 0; k < 10; k++) {
					cout << aaa[i][j][k] << " ";
				}
				cout << endl;
			}
			cout << "========================================" << endl;
		}
		cout << "++++++++++++++++++++++++++++++++++++++++++++++" << endl;
	}
}