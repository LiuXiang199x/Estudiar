#include "reference.h"

using namespace std;


int Reference::simple_test(){

    int a = 99;
    int &r = a;
    cout << a << ", " << r << endl;
    cout << &a << ", " << &r << endl;
    //99, 99, 0x28ff44, 0x28ff44

    printf("============================\n");
    int b = 99;
    int &rr = b;
    rr = 47;
    cout << b << ", " << rr << endl;    // 47 47; r和a是连体的，谁变都会改变对方
    
    return 0;
}

int Reference::test_reference(){


    return 0;
}