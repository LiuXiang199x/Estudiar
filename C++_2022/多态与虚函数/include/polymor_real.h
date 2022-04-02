#ifndef _POLYMOR_REAL_H_
#define _POLYMOR_REAL_H_
#include "allLibs.h"

int test_polyreal();

//基类People
class People{
public:
    People(char *name, int age);
    virtual void display();
protected:
    char *m_name;
    int m_age;
};

//派生类Teacher
class Teacher: public People{
public:
    Teacher(char *name, int age, int salary);
    virtual void display();
private:
    int m_salary;
};

#endif // !_POLYMOR_REAL_H_
