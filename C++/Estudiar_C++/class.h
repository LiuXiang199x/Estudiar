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
		cout << x.c << endl;  // �����ڳ�ʼ������ֵ��û��ֵ�Ļ������Ҹ�����������Ͳ��������������
		// cout << x.bb[0] << endl;  // ��仰�ᱨ��x.bb[0],��Ϊ0��1λ�ø���û�ж���
	}
};