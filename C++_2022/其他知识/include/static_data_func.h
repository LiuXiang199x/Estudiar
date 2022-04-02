#ifndef _STATIC_DATAFUNC_H_
#define _STATIC_DATAFUNC_H_
#include "allLibs.h"

class AA{
    public:
    int a=1;
    static int b;
    
    void print1();
    static void print2();
};

int test_static();

#endif // !_STATIC_DATAFUNC_H_