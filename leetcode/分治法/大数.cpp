#include<iostream>

using namespace std;


void RandomBigNumFun(int low, int high){

	unsigned int temp[1024] = {0};
	temp[0] = low;
	int flag = -1;
	unsigned int m_count = 1;
	int _index, index;

	for(_index=0; _index<high-1; ++_index){
		for(index=0; index<m_count; ++index){
			temp[index] *= low;

			if(flag!=-1){
				temp[index] += flag;
				flag = -1;
			}

			if(temp[index]>9999){
				flag = temp[index]/10000;
				temp[index]%=10000;
			}
		}
		if(flag!=-1){
			temp[index]+=flag;
			++m_count;
			flag=-1;
		}

		if(m_count > 1023){
			printf("数据过大而数组过小，请重置存放数组的大小");
			exit(0);
		}
	}

	for(index=m_count-1; index>=0; --index){
		if(temp[index]<10)
			cout << "000" << temp[index];
		else if(temp[index]<100)
			cout << "00" << temp[index];
		else if(temp[index]<1000)
			cout << "0" << temp[index];
		else
			cout << temp[index];
	
	}
}


int main(int argc, char **argv){

	int low_num, high_num;
	cout << argv[0] << endl; 
	cout << "type of argv[1]:" << typeid(argv[1]).name() << endl;
	cout << "type of *argv[2]:" << typeid(*argv[2]).name() << endl;
	cout << "low number = ";
	cin >> low_num;
	cout << "high number = ";
	cin >> high_num;

	RandomBigNumFun(low_num, high_num);

	return 0;
}
