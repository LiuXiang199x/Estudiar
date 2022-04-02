#ifndef _POLYMORPHISM_H_
#define _POLYMORPHISM_H_
#include "allLibs.h"


// 构造函数可以写在类中： XXX(); -> 然后再去cpp里面初始化构造函数
// 也可以写成：XXX(): ..... {}

int test_polymorphism();

class Shape {
    protected:
        int width, height;
        
    public:
        Shape(int aa=0, int bb=0);
        virtual int area();
};

class Rectangle: public Shape{
    public:
        // 继承, 派生类继承基类的构造函数（继承某种构造方法）
        // 这里函数里面其实已经不用再详细写了，因为Shape::Shape里面帮你把内容定义好了
        // 不用去cpp里面写 XXX::XXX(){....}
        Rectangle(int a=1, int x=2):Shape(a, x){}
        int area ();
};

class Triangle: public Shape{
    public:

        Triangle( int a=3, int b=4):Shape(a, b) { }
        int area ();
};

#endif // !_POLYMORPHISM_H_
