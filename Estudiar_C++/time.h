#pragma once
#include "alllib.h"

// if param is "NULL"

void paramNULL() {
	time_t current_time;
	current_time = time(NULL);
	printf("%d\n", current_time);
}

// ����������NULLʱ������������һ��ָ��time_t���͵�ʵ�壨��������е�test_time����һ��ָ�루��������е�p����
// ��ʱ����time()�ķ���ֵ��Ȼ�Ǵ�1970��1��1��������������ʱ�䣨����Ϊ��λ������ͬ������η���ֵͬʱҲ������Ϊ������ָ�루p����ָ���ʵ�壨test_time��


void paramNotNull() {
	time_t current_time;
	time_t test_time;
	time_t* p;
	p = &test_time;
	current_time = time(NULL);
	printf("%d\n", current_time);
	Sleep(1000);
	current_time = time(NULL);
	current_time = time(p);//ͬʱ��ֵ 
	printf("%d\n", current_time);
	printf("%d\n", test_time);
}
