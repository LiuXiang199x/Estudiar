#pragma once
#include"alllib.h"

using namespace std;

namespace value {
	void test() {
		int a = 1, b = 2;
		cout << "a=" << a << ",b=" << b << endl;
		swap(a, b);
		cout << "a=" << a << ",b=" << b << endl;
	}

	void swap(int x, int y)
	{
		int p = x;
		x = y;
		y = p;
	}
}


namespace pointer {
	void swap_error(int* x, int* y);
	void swap(int* x, int* y);

	void test() {
		int a = 1, b = 2;
		cout << "a=" << a << ",b=" << b << endl;
		cout << "&a=" << &a << ",&b=" << &b << endl;
		cout << "********1**********" << endl;

		cout << "normal pointer swap:" << endl;
		swap(&a, &b);
		cout << "a=" << a << ",b=" << b << endl;
		cout << "&a=" << &a << ",&b=" << &b << endl;
		cout << "********3**********" << endl;

		cout << "error pointer swap:" << endl;
		swap_error(&a, &b);
		cout << "a=" << a << ",b=" << b << endl;
		cout << "&a=" << &a << ",&b=" << &b << endl;
		cout << "********3**********" << endl;

	}

	void swap(int* x, int* y)
	{
		int p = *x;   //int p; p=*x;
		*x = *y;
		*y = p;
		cout << "x=" << x << ",y=" << y << endl;
		cout << "*x=" << *x << ",*y=" << *y << endl;
		cout << "&x=" << &x << ",&y=" << &y << endl;
		cout << "********2**********" << endl;
	}

	void swap_error(int* x, int* y)
	{
		int *p = x;   //int p; p=*x;
		x = y;
		y = p;
		cout << "x=" << x << ",y=" << y << endl;
		cout << "*x=" << *x << ",*y=" << *y << endl;
		cout << "&x=" << &x << ",&y=" << &y << endl;
		cout << "********2**********" << endl;
	}
}

namespace reference {

	void test();
	void swap(int& x, int& y);
	void test()
	{
		int a = 1, b = 2;
		cout << "a=" << a << ",b=" << b << endl;
		cout << "&a=" << &a << ",&b=" << &b << endl;
		cout << "*******1**********" << endl;
		swap(a, b);
		cout << "a=" << a << ",b=" << b << endl;
		cout << "&a=" << &a << ",&b=" << &b << endl;
		cout << "*******4**********" << endl;
	}
	void swap(int& x, int& y)
	{
		cout << "x=" << x << ",y=" << y << endl;
		cout << "&x=" << &x << ",&y=" << &y << endl;
		cout << "*******2**********" << endl;
		int p = x;
		x = y;
		y = p;
		cout << "x=" << x << ",y=" << y << endl;
		cout << "&x=" << &x << ",&y=" << &y << endl;
		cout << "*******3**********" << endl;
	}

	void test_reference() {
		int a = 88;
		//��������a��һ������c��c�Ǳ���a��һ������,������ã�������ʱ��һ��Ҫ��ʼ��
		int& c = a;
		//����������ʱ��һ��Ҫ��ʼ����һ�����������ж������
		int& d = a;
	
		cout << "a=" << a << endl;
		cout << "c=" << c << endl;
		cout << "=====================" << endl;
		c = 99;
		cout << "a=" << a << endl;
		cout << "========= ���ú͵�ַ���� =======" << endl;
		cout << "&(����)==>������ֵ�������ڱ������������λ�ڱ��� ���ʱ,��ʾ������������." << endl;
		cout << "&(ȡ��ַ�����)==>������ȡ�׵�ַ��"
			<< "�ڸ���������ֵʱ�����ڵȺ��ұ߻���ִ���������ΪһԪ���������ʱ��ʾȡ����ĵ�ַ." << endl;
		cout << "��������һ��������ã��ͱ�����һ�����ȡַ" << endl;
		cout << endl;
	
		cout << "======== ���ú�ָ������� ========" << endl;
		cout << "1.���ȣ����ò�����Ϊ�գ���ָ�����Ϊ�ա�" << endl;
		cout << "2.���ò����Ըı�ָ�򣬶�һ������\"��������\"������ָ����Ըı�ָ�򣬶�ָ����������" << endl;
		cout<<"3.�ٴΣ����õĴ�С����ָ��ı����Ĵ�С����Ϊ����ֻ��һ���������ѣ�ָ����ָ�뱾��Ĵ�С��4���ֽ�"<<endl;
		cout << "4.������ñ�ָ�����ȫ�����ڲ����ڿ����ã���������һ������ʼ��Ϊָ��һ���������Ͳ��ܱ��ı�Ϊ��һ����������ã�������úܰ�ȫ��" << endl;
		cout << "һ�仰����Ϊ���ǣ�ָ��ָ��һ���ڴ棬������������ָ�ڴ�ĵ�ַ������������ĳ���ڴ�ı��������ò��ı�ָ��" << endl;
	}
}