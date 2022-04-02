#include "switch_case_label.h"

using namespace std;

int test_switch_case(){
    cout << "case就是一个特殊的label, case标签必须是整型常量表达式" << endl;

    enum color{red, green, yellow};

    cout << green << endl;
    int cc;
    cin >> cc;

    // case 不加break的话，你进入第一个case后，他会自动运行接下来所有case
    switch(cc)
    {
    case red: cout << "i am in red!!!===> " << red << endl; break;
    case green: cout << "i am in green!!!===> " << green << endl; break;
    case yellow: cout << "i am in yellow!!!===> " << yellow << endl; break;
    }

    return 0;
}

// XXX: 带个冒号的就是一个label， goto label就能直接跳到那一行
int test_label(){

    cout << "XXX: 带个冒号的就是一个label， goto label就能直接跳到那一行" << endl;
    int a = 0;
    for(int i=0; i<10; i++){
        a++;
        cout << "a=" << a <<endl;
        goto hhhh;
    }

    cout << "==========" << endl;
    hhhh:
    cout << "++++++++++" << endl;

    return 0;
}