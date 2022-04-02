#include<iostream>

using namespace std;


// struct是一种结构体：是由一系列具有相同类型或不同类型的数据构成的数据集合，叫做结构。是一种复合数据类型，结构类型。
// C语言中结构体不能有函数，而C++中可以包含函数
/*
struct tag
{
	member-list
}variable-list;
注：struct为结构体关键字；   tag为结构体的标志；   member-list为结构体成员变量列表，其必须列出其所有成员；   variable-list为此结构体声明的变量； tag、member-list、variable-list这3部分至少要出现2个。*/

struct node{
	
	int a;
	int b[10];
	char c;
	double d;

	int y(int p){
		return p+1;	
	}
	int z;
	void add(){
		z++;
	}
};  // 不要忘记分号！！

// struct的好处，可将同一类或者同一个用途的数据去分类
// 可以和  typedef struct连用，具体见typedef
int main(int argc, char **argv){

	node x;
	x.a = 10;
	x.b[1]++;
	x.c = 'c';
	x.d = 3.14;
	x.z = x.y(x.d);
	cout << "x.z=" << x.z << endl;
	cout << "type of x:" << typeid(x).name() << endl;
	x.add();
	cout << "x.z=" << x.z << endl;

	return 0;
}
