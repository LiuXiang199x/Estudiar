#include "arr_string_calc_pointer.h"

using namespace std;

int Pointer::simple_max_menos(){

    int a=10, b=20;

    int *p1=&a, *p2;

    p2 = &b;

    cout << " &a=" << &a << " p1=" << p1 << " *p1=" << *p1 << " &p1=" << &p1 << endl;
    cout << " &b=" << &b << " p2=" << p2 << " *p2=" << *p2 << " &p2=" << &p2 << endl;
    // int d = *p1++;  // 10
    // int dd = ++*p1; // 11
    // cout << dd << endl;

    // (*p)++: 后赋值自加，传给别人的还是*p（内容上操作）；
    // ++*p = ++*(p): 先赋值/自加，传给别人的是自加完了的（内容上操作）;
    // *++p = *(++p): 先加地址（地址上操作）;
    // *p++ = *(p++): 先*p操作然后再地址++(先值不变*p取出来后再p存放的地址++)；
    // 32位机器，每一位寻址就是一个字节，int占四个字节，所以地址+1就是8c-90. +1 或自增都是移动4位（int类型）

    cout << " *p1++=" << *p1++ << "先赋值*p1再++， *++p1=" << *++p1 << "地址加了一位再取值， ++*p1=" << ++*p1 << \
        "前面地址已经加了一位了，在内容中+1， *(p1++)=" << *(p1++) << endl;
    cout << *p1 << " " << p1 << endl;       // 1140850815 0x7ffee6acd94d
    cout << *(p1--) << " " << p1 << endl;   // 1140850815 0x7ffee6acd949
    cout << *p1 << " " << p1 << endl;       // -18436903  0x7ffee6acd949
    cout << *(++p1) << " " << p1 << endl;   // 1140850815 0x7ffee6acd94d
    cout << *p1 << " " << p1 << endl;       // 1140850815 0x7ffee6acd94d
    cout << *++p1 << " " << p1 << endl;   // 1140850815 0x7ffee6acd94d
    cout << *p1 << " " << p1 << endl;       // 1140850815 0x7ffee6acd94d
    // 重点！！注意上面的dd，++*p1 到了11，cout里面又经过了一次++p1，到这个位置的时候dd已经位12了。
    // cout << dd << endl;   // 12

    int *ppp;
    int ii=2;
    ppp = &ii;
    // 0x7ffeb445be8c 0x7ffeb445be90 0x7ffeb445be8c 0x7ffeb445be90
    // 32位机器，每一位寻址就是一个字节，int占四个字节，所以地址+1就是8c-90.
    // ppp+1 并不是自增，所以看后面可以知道ppp++打印还是原来ppp地址; 若改为++ppp就是打印的自增后的地址
    // cout << ppp << " " << ppp+1 << " " << ++ppp << " " << ppp << endl;

    // 2 0x7ffc19c7bbcc 2 0x7ffc19c7bbd0 432520148
    cout << *ppp << " " << ppp << " " << ++*ppp << " " << ppp << " " << *ppp << endl;
    return 0;
}

int Pointer::array_pointer(){

    int *p, *pp;
    int arr[4] = {1,3,5,7};

    p = arr;
    pp=arr;
    cout << *(arr+1) << endl;
    cout << *p << " " << *(p+2) << endl;
    cout << arr << endl; // 0x7fff60688cd0
    
    // 1, 3, 4, 4: *pp取完1后地址++，
    // 1 - 0x7fff60688cd4, 3 - 0x7fff60688cd4, 
    // 4 - 0x7fff60688cd4(这里说明存的是3的地址，值上面+1)
    // 重点！：：！ 4 - 0x7fff60688cd4，这里存4 的原因！！前面本来这个地址是放的3，自加后这个地址的值被覆盖了！！
    cout << *pp++ << " " << pp << \
    " " << *pp << " " << pp << " "<< ++*pp << " "<<pp<<" " << *pp << " " << pp << endl;


    return 0;
}

int Pointer::string_pointer(){

    cout << "==========> C中有两种字符串表达方式：字符数组 和 指针直接指向字符串" << endl;
    cout << "==========> 字符数组（数据储存在全局静态区或栈区，能读能写）：" << endl;

    char str[] = "wo shi liu xiang";
    int len = strlen(str);

    cout << str << " " << len << endl;

    char *pstr = str;
    for(int i=0; i<len; i++){
        printf("%c", *(pstr+i));
    }
    printf("\n");

    cout << "==========> 指针直接指向字符串（字符串储存在常量区, 能读不能写）：" << endl;

    char *pstrr = "me llamo marco....";
    // char *pstrr; pstrr = ".....";
    cout << pstrr << endl;
    int len_s = strlen(pstrr);
    for(int j=0; j<len_s; j++){
        printf("%c", *(pstrr+j));
    }
    printf("\n");

    return 0;
}
