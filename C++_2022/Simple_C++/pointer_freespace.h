#pragma once
#include"alllib.h"

using namespace std;

void Pointer() {

	cout << "计算机在存储数据时必须跟踪三种属性：" << endl;
	cout << "   1. 信息存储在何处" << endl;
	cout << "   2. 存储的值为多少" << endl;
	cout << "   3. 存储的信息是什么类型" << endl;
	cout << "指针策略是C++内存管理编程理念的核心" << endl;
	cout << endl;

	cout << "处理存储数据的新策略：将地址视为指定的量，将值视为派生量。一种特殊变量——指针" << endl;
	cout << "正常使用就不写了，写一些新学的" << endl;
	cout << "int *p; *p=5; p我们没有让他指向一个具体地址，而是直接赋值了，这个时候系统会随机分配，乱指。" << endl;

	cout << "指针真正的用处在于：在运行阶段分配未命名的内存以存储值" << endl;
}

void neww() {
	cout << "C语言中用malloc()来分配内存，C++也可以，但是C++还可以用new运算符！" << endl;
	int temp;
	int* p1 = &temp;
	int* p2 = new int;
	cout << "p1: int temp; int* p1 = &temp; ，p2:int* p2 = new int; 两种都属于初始化指针" << endl;

	int* p3 = new int;
	*p3 = 100;
	cout << p3 << " " << *p3;

	// 释放内存，delete只能删除new出来的内存
	// 因为new出来的内存是在堆heap或者自由储存区的
	// int出来的内存是在stack栈中的
	delete p3; //将内存还给内存池

	// new来创建动态数组
	int* psome = new int[10];
	delete[] psome; // []告诉delete是要释放整个数组占用的内存，psome只是指向第一位，否则就只释放了第一个位置

	// delete只能和new配套用
	// 不能delete同一块内存两次
	// delete数组注意带【】和不带的区别
	int* psome2 = new int[5];
	int arr2[2] = { 1,3 };
	psome2[0] = 0;
	psome2[1] = 1;
	psome2[2] = 2;
	psome2[3] = 3;
	psome2[4] = 4;
	cout << psome2 << endl;
	cout << psome2 + 0 << endl;
	cout << psome2 + 1 << endl;
	cout << psome2 + 2 << endl;
	cout << psome2 + 3 << endl;
	cout << psome2 + 4 << endl;
	cout << sizeof(psome2)<<" "<<sizeof(psome2[0]) << " " << sizeof(arr2) << endl;
	for (int i = 0; i < 5; i++) {
		cout << psome2[i] << endl;
		cout << *(psome2 + i) << endl;
		cout << &psome2[i] << endl;
	}
}

void pointer_array() {
	double wages[3] = { 10000.0, 20000.0, 30000.0 };
	short stacks[3] = { 3,2,1 };

	double* pw = wages;
	short* ps = &stacks[0];

	cout << "pw=" << pw << " " <<"*pw="<<*pw<<endl;
	pw = pw + 1;
	cout << "pw+1" << endl;
	cout << "pw=" << pw << " " << "*pw=" << *pw << endl;

	cout << "ps=" << ps << " " << "*ps=" << *ps << endl;
	ps = ps + 1;
	cout << "ps+1" << endl;
	cout << "ps=" << ps << " " << "*ps=" << *ps << endl;

	cout << "stacks[0]=" << stacks[0]
		<< ", stacks[1]=" << stacks[1] << endl;
	cout << "using *stacks:" << endl;
	cout << "*stacks=" << *stacks
		<< ", *(stacks+1)=" << *(stacks+1) << endl;
	cout << "sizeof(wages)=" << sizeof(wages);
	cout << ", sizeof(pw)=" << sizeof(pw);
	cout << ", sizeof(ps)=" << sizeof(ps);
	cout << ", sizeof(stacks)=" << sizeof(stacks) << endl;;
}

void pointer_strings() {
	char flower[10] = "rose";
	cout << flower << endl;
	const char* ps = "bear";
	cout << ps << " " << *ps << endl;
}