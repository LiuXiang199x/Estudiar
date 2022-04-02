#include "this_pointer.h"

using namespace std;


Point::Point(int xx, int yy){
    x = xx;
    y = yy;
}

void Point::MovePoint(int qq, int pp){
    x += qq;
    y += pp;
}

void Point::print(){
    cout << "x = " << x << \
    "; y = " << y << endl;
    cout << this->x << " " << this->y << endl;
}

int test_this(){
    Point point1(10, 10);
    cout << sizeof(Point) << " " << sizeof(point1) << endl;
    point1.MovePoint(2,2);

    Point* pstp = new Point(0, 0);
    pstp->MovePoint(33, 55);
    pstp->print();
    
    point1.print();
}