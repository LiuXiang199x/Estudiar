#include "static_data_func.h"

using namespace std;

void AA::print1(){
    cout << "i am in print1111 " << endl;
}

void AA::print2(){
    cout << "b:::" << b << endl;
}

int AA::b = 4;

int test_static(){
    
    // AA::b = 2;

    AA* p = new AA;
    cout << p->a << endl;
    cout << p->b << endl;
    p->b++;
    cout << p->b << endl;
    AA::print2();
    AA::b++;
    cout << AA::b << endl;
    p->print2();

    return 0;
}