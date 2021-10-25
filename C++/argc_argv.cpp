#include<iostream>

using namespace std;

// argc 是argument count的缩写表示传入main函数中的参数个数，包括这个程序本身
// argv 是 argument vector的缩写表示传入main函数中的参数列表，其中argv[0]表示这个程序的名字
// **argv 也可以写为 *argv[]
// 在上面两个名词的解释中提到“这个程序”，所谓的“这个程序”就是包含main函数的程序。
int main(int argc, char **argv){ //char *argv[]

	int a;
	a = 0;
	cout << typeid(argc).name() << endl;   // i
	cout << typeid(argv).name() << endl;   // PPc
	cout << typeid(a).name() << endl;   // PPc
	cout << argv[0] << endl;   // ./a.out
	cout << argv[1] << endl;   // 没东西
	
	// argc表示argv中参数的个数。
	printf("the number of command line parameters is %d!\n",argc );
	for (int i = 0; i < argc; ++i)
	{
		printf("argv[%d] is %s\n",i,argv[i]);
	}
	
	return 0;
}
