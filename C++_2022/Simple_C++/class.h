#pragma once
#include"alllib.h"

using namespace std;

class test {
public:
	int a;
	int b;
	typedef struct {
		int a;
		vector<int> b;
		vector<int> bb;
		float c;
	}Data;
	Data x;

	void cout_sum() {

		x.a = 2;
		x.b = { 1,2 };
		cout << a << " " << b << endl;
		cout << a + b << endl;
		cout << x.a << endl;
		cout << x.b[0] << endl;
		cout << x.c << endl;  // 所以在初始化单个值，没赋值的话他会乱给，但是数组就不会给，让它空着
		// cout << x.bb[0] << endl;  // 这句话会报错x.bb[0],因为0和1位置根本没有东西
	}
};