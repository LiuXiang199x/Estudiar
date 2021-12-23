#pragma once
#include"alllib.h"

using namespace std;

void test() {
	int a = 1;
	int b = 2;
	int c = 3;
	// 慎用else if，esle if是只有在if没有触发的情况下才会触发else if
	if (a == 1 && b == 2) {
		cout << "1 t" << endl;
	}
	if (a == 1 || b == 3) {
		cout << "2 t" << endl;
	}
	if (a == 1 && b == 3) {
		cout << "3t" << endl;
	}
	if (a == 1 && c == 3) {
		cout << "4t" << endl;
	}
}
