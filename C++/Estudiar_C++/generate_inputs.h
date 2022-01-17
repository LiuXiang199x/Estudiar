#pragma once
#include"alllib.h"
vector<vector<vector<double>>> get_inputs() {
	// srand(time(0)); ���Ҫ������Ҫ�ĳ�������ȥ��������Ȼÿ�ν������еõ������ݻ���������ɵ�һ����
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
	srand((unsigned)time(NULL));//time()��ϵͳʱ���ʼ���֡�Ϊrand()���ɲ�ͬ��������ӡ�
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