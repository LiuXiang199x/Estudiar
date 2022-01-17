#pragma once
#include"alllib.h"

int findMaxValue(vector<float> data) {
	float tmp = data[0];
	int max_index = 0;

	for (int i = 0; i < data.size(); i++) {
		if (tmp < data[i]) {
			tmp = data[i];
			max_index = i;
		}
	}
	return max_index;
}

void test() {
	vector<float> arr1 = {1.1, 1.2, 1.5, 1.4, 1.5, 1.2 ,1.1 ,1.3};

	cout << "Max value index is ==> " << findMaxValue(arr1) << endl;
}
