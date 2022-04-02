#include "pointer_func.h"

using namespace std;

void coutf(int *arr, int len){
    cout << len << endl;
    cout << *(arr+len-2) << endl;
    *(arr+2) = 1000;
}

char *strlong(char *str1, char *str2){
    if(strlen(str1)>strlen(str2)){
        return str1;
    }else{
        return str2;
    }
}

int pointer_func(){

    cout <<   "=========> 指针作为函数的传参：" << endl;
    int nums[5] = {1,2,3,4,5};
    int len = sizeof(nums) / sizeof(int);
    coutf(nums, len);
    for(int i=0; i<5; i++){
        cout << nums[i] << " ";
    }
    cout << endl;

    cout << "\n==========> 指针作为函数的返回值：" << endl;
    char str1[]="wo shi liu xiang", str2[]="wo jin nian 26 sui", *str;
    cout << str1 << "; " << str2 << endl;

    str = strlong(str1, str2);
    cout << str << endl;
    
    return 0;
}
