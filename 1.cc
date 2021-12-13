#include<iostream>
#include<string>


using namespace std;

int main(){
        char q[3] = {"qw"};
        string qq = "qwqw";
        cout <<q[0]<<" "<< q[-1] <<" "<<q[-2]<< endl;
	cout << qq.length() << endl;
        cout << qq[0] << " "<< qq[3] << " " <<
		"==="<<qq[-1] << " " << qq[-2] << endl;
        return 0;
}

