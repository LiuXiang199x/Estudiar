#pragma once
#include"alllib.h"

void arrays() {
	int intarray[3];
	int intarray2[4] = { 2,3,4,5 };
	int intarray3[] = { 1,1,1 };
	intarray[0] = 1;
	intarray[1] = 2;
	intarray[2] = 3;

	cout << intarray[1] << endl;
	for (int i = 0; i < sizeof(intarray2) / sizeof(intarray2[0]); i++) {
		cout << intarray2[i] << endl;
	}
	cout << intarray3 << endl;
}

void char_strings() {

	char dog[3] = { 'd','o','g' };
	char cat[4] = { 'c','a','t','\0' };
	char name[] = "liuxiang";   // 9, 说明后面默认计算了"\0"

	cout << typeid(dog).name() << endl;
	cout << typeid(name).name() << endl;
	cout << sizeof(name) / sizeof(name[0]) << endl;
}
