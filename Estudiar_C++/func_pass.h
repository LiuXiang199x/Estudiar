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
		//声明变量a的一个引用c，c是变量a的一个别名,如果引用，声明的时候一定要初始化
		int& c = a;
		//引用声明的时候一定要初始化，一个变量可以有多个引用
		int& d = a;
	
		cout << "a=" << a << endl;
		cout << "c=" << c << endl;
		cout << "=====================" << endl;
		c = 99;
		cout << "a=" << a << endl;
		cout << "========= 引用和地址区别 =======" << endl;
		cout << "&(引用)==>用来传值，出现在变量声明语句中位于变量 左边时,表示声明的是引用." << endl;
		cout << "&(取地址运算符)==>用来获取首地址，"
			<< "在给变量赋初值时出现在等号右边或在执行语句中作为一元运算符出现时表示取对象的地址." << endl;
		cout << "和类型在一起的是引用，和变量在一起的是取址" << endl;
		cout << endl;
	
		cout << "======== 引用和指针的区别 ========" << endl;
		cout << "1.首先，引用不可以为空，但指针可以为空。" << endl;
		cout << "2.引用不可以改变指向，对一个对象\"至死不渝\"；但是指针可以改变指向，而指向其它对象。" << endl;
		cout<<"3.再次，引用的大小是所指向的变量的大小，因为引用只是一个别名而已；指针是指针本身的大小，4个字节"<<endl;
		cout << "4.最后，引用比指针更安全。由于不存在空引用，并且引用一旦被初始化为指向一个对象，它就不能被改变为另一个对象的引用，因此引用很安全。" << endl;
		cout << "一句话归纳为就是：指针指向一块内存，它的内容是所指内存的地址；而引用则是某块内存的别名，引用不改变指向。" << endl;
	}
}