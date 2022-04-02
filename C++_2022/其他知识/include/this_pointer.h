#ifndef _THIS_POINTER_H_
#define _THIS_POINTER_H_
#include "allLibs.h"

class Point{

    public:
        Point(int a, int b);
        void MovePoint(int a, int b);
        void print();
    
    private:
        int x, y;
        char ch, i;
        double qqq;
};

int test_this();

#endif // !_THIS_POINTER_H_