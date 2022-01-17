#pragma once
#include"alllib.h"

#include"multiarrays.h"

void structs() {
	struct alumno {
		string name;
		int age;
		char sexo[];
	};

	alumno primero;
	alumno segundo = {
		"qeqeqw",
		24,
		"123qeqwe"
	};
	primero.name = "liuxiang";
	primero.age = 25;

	cout << segundo.name << endl;
	cout << primero.name << endl;
}

// 结构数组
void structs_array() {
	struct alumno {
		string name;
		int age;
	};

	alumno classA[2];
	alumno classB[2] = {
		{"xiaoming", 24},
		{"lihua", 23}
	};
	classA[0].name = "liuxiang";
	classA[0].age = 35;

	cout << classA[0].age << endl;
	cout << classB[1].name << endl;
}

struct maps_data {
	vector<vector<float>> OCmap;
	int map_x;
	int map_y;
};
maps_data return_struct() {
	maps_data map2;
	map2.OCmap = get_maps();
	map2.map_x = 2;
	map2.map_y = 2;
	return map2;
}
void cout_return_struct() {
	maps_data map;
	map = return_struct();
	cout << sizeof(map) << " " << sizeof(map.map_x) << " " << sizeof(map.map_y) << " " << sizeof(map.OCmap) << " " << typeid(map).name() << endl;
	cout << map.map_x << " " << map.map_y << endl;
	cout << typeid(map.OCmap).name() << endl;
}

class return_structs {
public:
	struct estudiante {
		int age;
		int year;
	};

	estudiante get_info() {
		estudiante ss;
		ss.age = 12;
		ss.year = 110;
		return ss;
	}

	void cout_info() {
		estudiante ssd;
		ssd.age = get_info().age;
		ssd.year = get_info().year;

		cout << ssd.age << " " << ssd.year << endl;
	}
};