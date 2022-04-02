#include "pointer_arr_doublearr.h"

using namespace std;

// 如果一个数组中的所有元素保存的都是指针，那么我们就称它为指针数组
// dataType *arrayName[length];
int pointer_arr(){

    int a=1, b=2, c=3;

    // 指针数组
    int *arr[] = {&a, &b, &c};

    int **parr = arr;
    printf("%d, %d, %d\n", *arr[0], *arr[1], *arr[2]);
    printf("%d, %d, %d\n", **(parr+0), **(parr+1), **(parr+2));

    cout << "=======> 字符串: " << endl;
    // 字符串指针。str是一个指针数组【pointer....】,里面存的都是指向字符串的指针
    // 所以数组名指向第一个元素是一个指针，所以数组名其实是一个二级指针
    char *str[3] = {
        "c.biancheng.net",
        "C语言中文网",
        "C Language"
    };
    printf("%s\n%s\n%s\n", str[0], str[1], str[2]);
    cout << str << endl;  // 0x7ffc2e5bf8b0
    cout << &str[0] << endl;  // 0x7ffc2e5bf8b0
    cout << *str << endl;  // c.biancheng.net

    char *str0 = "c.biancheng.net";
    char *str1 = "C语言中文网";
    char *str2 = "C Language";
    char *strr[3] = {str0, str1, str2};
    printf("%s\n%s\n%s\n", strr[0], strr[1], strr[2]);

    char arry[7][10] = {"Monday","Tuesday","Wednsday","Thurday","Friday","Saturday","Sunday"};

    return 0;
}

int pointer_doublearr(){
    int a[3][4] = { {0, 1, 2, 3}, {4, 5, 6, 7}, {8, 9, 10, 11} };
    int (*p)[4] = a;
    printf("%d\n", sizeof(*(p+1)));

    int aa[3][4]={0,1,2,3,4,5,6,7,8,9,10,11};
    int(*pp)[4];
    cout <<*pp<<endl; // 0x55c6239d3008
    cout << *++pp << endl;   // 0x55c6239d3018
    cout << *(*pp+1) << endl;  // 0
    int i,j;
    pp=aa;
    for(i=0; i<3; i++){
        for(j=0; j<4; j++) printf("%2d  ",*(*(pp+i)+j));
        printf("\n");
    }
    return 0;
}