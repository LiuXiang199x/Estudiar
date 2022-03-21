#include "SceneNet.h"

/*
    one_hot_selected_objects[0] = one_hot_all_objects[0]
    one_hot_selected_objects[1] = one_hot_all_objects[2]
    one_hot_selected_objects[2] = one_hot_all_objects[4]
    one_hot_selected_objects[3] = one_hot_all_objects[5]
    one_hot_selected_objects[4] = one_hot_all_objects[6]
    one_hot_selected_objects[5] = one_hot_all_objects[9]
    one_hot_selected_objects[6] = one_hot_all_objects[11]
    one_hot_selected_objects[7] = one_hot_all_objects[15]
    one_hot_selected_objects[8] = one_hot_all_objects[16]
    one_hot_selected_objects[9] = one_hot_all_objects[17]
    one_hot_selected_objects[10] = one_hot_all_objects[18]
    one_hot_selected_objects[11] = one_hot_all_objects[21]
    one_hot_selected_objects[12] = one_hot_all_objects[23]

    ['bed_room', 'dining_room', 'drawing_room', 'others', 'toilet']
*/

/*
["bed"-0, "cabinet"-1, "chair_base"-2, "cupboard"-3, "dining_table"-4, 
"door_anno"-5, "refrigerator"-6, "sofa"-7, "tea_table"-8, "toilet"-9, 
"TV_stand"-10, "metal_chair_foot"-11, "washing_machine"-12]
*/

using namespace std;

int main()
{
	float tmp_Nano[13] = {1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
	
    // change it to Mat
    // float tmp_Scene[512] = {1};
    cv::Mat img = cv::imread(img_path);
    
    everest::ai::SceneNet a;

    a.runSceneNet(img, tmp_Nano);

	return 0;
} 
