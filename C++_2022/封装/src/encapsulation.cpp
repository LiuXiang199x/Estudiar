#include "encapsulation.h"

using namespace std;

Adder::Adder(int i){
    total = i;
}

void Adder::addNum(int number){
    total = total + number;
}

int Adder::getTotal(){
    return total;
}