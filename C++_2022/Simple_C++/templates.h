#pragma once
#include"alllib.h"

namespace complex_func_same_meaning {
    //交换 int 变量的值
    void Swap(int* a, int* b) {
        int temp = *a;
        *a = *b;
        *b = temp;
    }
    //交换 float 变量的值
    void Swap(float* a, float* b) {
        float temp = *a;
        *a = *b;
        *b = temp;
    }
    //交换 char 变量的值
    void Swap(char* a, char* b) {
        char temp = *a;
        *a = *b;
        *b = temp;
    }
    //交换 bool 变量的值
    void Swap(bool* a, bool* b) {
        char temp = *a;
        *a = *b;
        *b = temp;
    }
}

// 这些函数虽然在调用时方便了一些，但从本质上说还是定义了三个功能相同、函数体相同的函数，只是数据的类型不同而已，这看起来有点浪费代码
// template 可以将他们集成为一个模板

namespace template_func {
    // 值（Value）和类型（Type）是数据的两个主要特征，它们在C++中都可以被参数化。
    // 搞个模板,建立一个通用函数，该函数用到的数据类型可不具体指定
 
    template<typename T> void Swap_pointer(T *a, T *b) {
        T temp = *a;
        *a = *b;
        *b = temp;
    }

    template<class T> void Swap_ref(T& a, T& b) {
        T temp = a;
        a = b;
        b = temp;
    }

    void test() {
        int n1 = 1, n2 = 0;
        cout << "Int>>>before swap: n1 = " << n1 << "; n2 = " << n2 << endl;
        Swap_pointer(&n1, &n2);
        cout << "Int>>>after swap: n1 = " << n1 << "; n2 = " << n2 << endl;
        cout << endl;

        float f1 = 1.0, f2 = 0.1;
        cout << "Float>>>before swap: f1 = " << f1 << "; f2 = " << f2 << endl;
        Swap_ref(f1, f2);
        cout << "Float>>>after swap: f1 = " << f1 << "; f2 = " << f2 << endl;
    }
}

namespace template_class {
/*
template<typename 类型参数1 , typename 类型参数2 , …> class 类名{
    //TODO:
};
*/
    template<typename T1, typename T2>  //这里不能有分号
    class Point {
    public:
        Point(T1 x, T2 y) : m_x(x), m_y(y) { }
    public:
        T1 getX() const;  //获取x坐标
        void setX(T1 x);  //设置x坐标
        T2 getY() const;  //获取y坐标
        void setY(T2 y);  //设置y坐标
    private:
        T1 m_x;  //x坐标
        T2 m_y;  //y坐标
    };
    template<typename T1, typename T2>  //模板头
    T1 Point<T1, T2>::getX() const /*函数头*/ {
        return m_x;
    }
    template<typename T1, typename T2>
    void Point<T1, T2>::setX(T1 x) {
        m_x = x;
    }
    template<typename T1, typename T2>
    T2 Point<T1, T2>::getY() const {
        return m_y;
    }
    template<typename T1, typename T2>
    void Point<T1, T2>::setY(T2 y) {
        m_y = y;
    }

    void test() {
        Point<int, int> p1(10, 20);
        cout << "x=" << p1.getX() << ", y=" << p1.getY() << endl;

        Point<int, float> p2(10, 15.5);
        cout << "x=" << p2.getX() << ", y=" << p2.getY() << endl;
    }
}