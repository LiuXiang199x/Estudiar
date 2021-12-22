#include <iostream>
#include <stdlib.h>
#include <time.h> 
using namespace std; 
int main()
{

	int a;
	a = time(0);
	cout << "a:" << a << endl;

	srand(time(0));

	for(int i = 0; i < 10;i++ ){ 
        	cout << rand() << endl; 
		cout << rand()%10 << endl;
	}
	
	for(int i=0; i<10; i++){
		cout << (rand()%10001)/10000.0 << endl;
	}
	cout << endl; 
	
	
	
	
	return 0;
}
