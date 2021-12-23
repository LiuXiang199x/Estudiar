#pragma once
#include"alllib.h"

void arrays() {
	vector<int> map(10, 0);
	cout << "first dim map size:" << map.size() << endl;

	vector<vector<int>> map2(10, vector<int>(10, 0));
	cout << "second dim map size:" << map.size() << " " << map2[0].size() << endl;

	// vector方式一
	vector<vector<vector<int>>> map3(3); // 创建3个vector<vector<int>>
	for (int n = 0; n < 3; n++) {
		map3[n].resize(4); //给第二维设置大小
		for (int i = 0; i < 2; i++) {    // 一定要先设置高维度，再来分配低纬度的内存，否则会报错的
			map3[n][i].resize(4);  //给第三维设置大小
		}
	}
	cout << "third dim map size:" << map3.size() << " " << map3[0].size() << " " << map3[0][0].size() << endl;

	// vector方式二
	vector<vector<vector<int>>> vec3(2, vector<vector<int>>(3, vector<int>(4)));
	cout << "third vector size:" << vec3.size() << " " << vec3[0].size() << " " << vec3[0][0].size() << endl;

}

vector<vector<float>> get_maps() {
	vector<vector<float>> map(4, vector<float>(4, 1.2));
	int r_pos_x = 10;
	int r_pos_y = 10;
	return map;
}

void cout_maps() {
	vector<vector<float>> map;
	map= get_maps();
	float a;
	double b;
	cout << sizeof(b) << endl;
	cout << sizeof(map) << " " << sizeof(map[0]) << " " << sizeof(map[0][0]) << endl;
	for (int i = 0; i < sizeof(map) / sizeof(map[0][0]); i++) {
		for (int j = 0; j < sizeof(map[0]) / sizeof(map[0][0]); j++) {
			cout << map[i][j] << " ";
		}
		cout << endl;
	}
}

void vector_test() {
	vector<vector<int>> a(3, vector<int>(3, 0));
	vector<vector<int>> b(3, vector<int>(3, 1));
	cout << a[0][0] << " " << a[1][1] << " " << b[0][0] << " " << b[1][1] << endl;
}

double cout_arr(double* p) {
	cout << *p << " " << *(p + 1)<<endl;
	*p = 1;
	double q[2];
	q[0] = *p;
	q[1] = *(p + 1);
	return *q;
}