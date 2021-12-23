#pragma once
#include"alllib.h"

namespace complex_func_same_meaning {
    //���� int ������ֵ
    void Swap(int* a, int* b) {
        int temp = *a;
        *a = *b;
        *b = temp;
    }
    //���� float ������ֵ
    void Swap(float* a, float* b) {
        float temp = *a;
        *a = *b;
        *b = temp;
    }
    //���� char ������ֵ
    void Swap(char* a, char* b) {
        char temp = *a;
        *a = *b;
        *b = temp;
    }
    //���� bool ������ֵ
    void Swap(bool* a, bool* b) {
        char temp = *a;
        *a = *b;
        *b = temp;
    }
}

// ��Щ������Ȼ�ڵ���ʱ������һЩ�����ӱ�����˵���Ƕ���������������ͬ����������ͬ�ĺ�����ֻ�����ݵ����Ͳ�ͬ���ѣ��⿴�����е��˷Ѵ���
// template ���Խ����Ǽ���Ϊһ��ģ��

namespace template_func {
    // ֵ��Value�������ͣ�Type�������ݵ�������Ҫ������������C++�ж����Ա���������
    // ���ģ��,����һ��ͨ�ú������ú����õ����������Ϳɲ�����ָ��
 
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
template<typename ���Ͳ���1 , typename ���Ͳ���2 , ��> class ����{
    //TODO:
};
*/
    template<typename T1, typename T2>  //���ﲻ���зֺ�
    class Point {
    public:
        Point(T1 x, T2 y) : m_x(x), m_y(y) { }
    public:
        T1 getX() const;  //��ȡx����
        void setX(T1 x);  //����x����
        T2 getY() const;  //��ȡy����
        void setY(T2 y);  //����y����
    private:
        T1 m_x;  //x����
        T2 m_y;  //y����
    };
    template<typename T1, typename T2>  //ģ��ͷ
    T1 Point<T1, T2>::getX() const /*����ͷ*/ {
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