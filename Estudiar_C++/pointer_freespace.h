#pragma once
#include"alllib.h"

using namespace std;

void Pointer() {

	cout << "������ڴ洢����ʱ��������������ԣ�" << endl;
	cout << "   1. ��Ϣ�洢�ںδ�" << endl;
	cout << "   2. �洢��ֵΪ����" << endl;
	cout << "   3. �洢����Ϣ��ʲô����" << endl;
	cout << "ָ�������C++�ڴ����������ĺ���" << endl;
	cout << endl;

	cout << "����洢���ݵ��²��ԣ�����ַ��Ϊָ����������ֵ��Ϊ��������һ�������������ָ��" << endl;
	cout << "����ʹ�þͲ�д�ˣ�дһЩ��ѧ��" << endl;
	cout << "int *p; *p=5; p����û������ָ��һ�������ַ������ֱ�Ӹ�ֵ�ˣ����ʱ��ϵͳ��������䣬��ָ��" << endl;

	cout << "ָ���������ô����ڣ������н׶η���δ�������ڴ��Դ洢ֵ" << endl;
}

void neww() {
	cout << "C��������malloc()�������ڴ棬C++Ҳ���ԣ�����C++��������new�������" << endl;
	int temp;
	int* p1 = &temp;
	int* p2 = new int;
	cout << "p1: int temp; int* p1 = &temp; ��p2:int* p2 = new int; ���ֶ����ڳ�ʼ��ָ��" << endl;

	int* p3 = new int;
	*p3 = 100;
	cout << p3 << " " << *p3;

	// �ͷ��ڴ棬deleteֻ��ɾ��new�������ڴ�
	// ��Ϊnew�������ڴ����ڶ�heap�������ɴ�������
	// int�������ڴ�����stackջ�е�
	delete p3; //���ڴ滹���ڴ��

	// new��������̬����
	int* psome = new int[10];
	delete[] psome; // []����delete��Ҫ�ͷ���������ռ�õ��ڴ棬psomeֻ��ָ���һλ�������ֻ�ͷ��˵�һ��λ��

	// deleteֻ�ܺ�new������
	// ����deleteͬһ���ڴ�����
	// delete����ע��������Ͳ���������
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