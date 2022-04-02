#include "const.h"

using namespace std;

int getNum(){
    return 100;
}

int test_const(){

    const int MaxNum = 100;
    // MaxNum = 10;  // 会报错：表达式必须是可修改的左值
    int n = 90;
    const int MaxNum1 = getNum();  //运行时初始化
    const int MaxNum2 = n;  //运行时初始化
    const int MaxNum3 = 80;  //编译时初始化
    printf("%d, %d, %d\n", MaxNum1, MaxNum2, MaxNum3);


    return 0;
}

int const_pointer(){

    // const 也可以和指针变量一起使用，这样可以限制指针变量本身，也可以限制指针指向的数据。
    // const 离变量名近就是用来修饰指针变量的，离变量名远就是用来修饰指针指向的数据，
    // 如果近的和远的都有，那么就同时修饰指针变量以及它指向的数据。
    
    cout << "const type* | type const* 都是常量指针，指针指向的数据在常量区不可改变，指针可以改变指向的东西" << endl;
    cout << "type * const 是指针常量，指针指向的数据可改变可通过指针改变，指针不可以改变指向的东西" << endl;
    const int* p1;   // 指针指向的数据都只是可读，但是p1指针本身可以修改，它指向的数据不能修改
    int const* p2;   // 指针指向的数据只是可读，但是p2指针本身可以修改，它指向的数据不能修改

    int a =4; 
    int * const p3 = &a;

    return 0;
}

// 统计出现了几次
size_t strnchrr(const char *str, char *ch){
    int i, n = 0, len = strlen(str);
    // cout << "strlen(len): " << len << endl;
    for(i=0; i<len; i++){
        // cout << str[i] << endl;
        if(str[i] == *ch){
            n++;
        }
    }
    return n;
}

int const_funcParams(){

    cout << "单独定义 const 变量没有明显的优势，完全可以使用#define命令代替。" << endl;
    cout << "const 通常用在函数形参中，如果形参是一个指针，为了防止在函数内部修改指针指向的数据，就可以用 const 来限制" << endl;

    char const *info = "wo shi liu xiang";
    char ch[] = "i";
    // char *ch2;
    // strcpy(ch2, "wo jin nian 26 sui");
    // cout << strlen(ch2) << endl;

    int n = strnchrr(info, ch);  // 从字符串str中寻找字符character第一次出现的位置。
    printf("%d\n", n);


    return 0;
}
