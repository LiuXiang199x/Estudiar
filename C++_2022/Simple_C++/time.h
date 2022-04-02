#pragma once
#include "alllib.h"

// if param is "NULL"

void paramNULL() {
	time_t current_time;
	current_time = time(NULL);
	printf("%d\n", current_time);
}

// 当参数不是NULL时，参数必须是一个指向time_t类型的实体（下面代码中的test_time）的一个指针（下面代码中的p），
// 此时函数time()的返回值仍然是从1970年1月1日至今所经历的时间（以秒为单位），不同的是这次返回值同时也赋给作为参数的指针（p）所指向的实体（test_time）


void paramNotNull() {
	time_t current_time;
	time_t test_time;
	time_t* p;
	p = &test_time;
	current_time = time(NULL);
	printf("%d\n", current_time);
	Sleep(1000);
	current_time = time(NULL);
	current_time = time(p);//同时赋值 
	printf("%d\n", current_time);
	printf("%d\n", test_time);
}
