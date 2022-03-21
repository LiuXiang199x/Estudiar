/**********************************************************************************
File name:	  ai.h
Author:       Kimbo
Version:      V1.6.2
Date:	 	  2016-10-27
Description:  None
Others:       None

History:
	1. Date:
	Author: Kimbo
	Modification:
***********************************************************************************/

#ifndef EVEREST_AI_AI_H
#define EVEREST_AI_AI_H
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include <iostream>

//#define CAP_IMAGE
namespace everest
{
    namespace ai
    {
        enum TAIObjectClass
        {
            AI_OBJECT_NOTHING      = 999,
            AI_OBJECT              = 1000, // object flag
            AI_OBJECT_SOCKS        = 1001,
            AI_OBJECT_SLIPPER      = 1002,    //ng
            AI_OBJECT_WIRE         = 1003,
            AI_OBJECT_CHAIR_BASE   = 1004,     //ng
            AI_OBJECT_BLANKET      = 1005,     //ng
            AI_OBJECT_DOG          = 1006,
            AI_OBJECT_CAT          = 1007,
            AI_OBJECT_ANIMAL_FOOD  = 1008,    //na
            AI_OBJECT_POTATO_CHIPS = 1009,    //na
            AI_OBJECT_SEED_SHELL   = 1010,    //na
            AI_OBJECT_DROPPINGS    = 1011,
            AI_OBJECT_SHOE         = 1012,
            AI_OBJECT_FOOT         = 1013,    //na
            AI_OBJECT_PEOPLE       = 1014, 
            AI_OBJECT_PEOPLE_LAID  = 1015,    //na
            AI_OBJECT_DIRT         = 1016,    //ng
            AI_OBJECT_WEIGHT_SCALE = 1017,
            AI_OBJECT_CHARGER      = 1018,
            AI_OBJECT_METAL_CHAIR_FOOT  = 1038,

            AI_OBJECT_DIRTY_FLOOR  = 1039,
            AI_OBJECT_DIRTY_LIQUID  = 1040,
            AI_OBJECT_DIRTY_SOLID  = 1041,
            AI_OBJECT_DIRTY_HAIRE  = 1042,
            AI_OBJECT_DIRTY_OBJIECT  = 1043,
            

            AI_OBJECT_UKNOWN_SOCKS       = 1101,
            AI_OBJECT_UKNOWN_WEIGHT_SCALE   = 1117,
            AI_OBJECT_UKNOWN_CHAIR_BASE     = 1104,
            AI_OBJECT_UKNOWN_METAL_CHAIR_FOOT   = 1138,
            AI_OBJECT_UKNOWN_SHOE        = 1112,
            AI_OBJECT_UKNOWN_WIRE        = 1103,
            

            AI_OBJECT_FURNITURE    = 1510, // room object flag
            AI_OBJECT_DINING_TABLE = 1511,
            AI_OBJECT_SOFA         = 1512,
            AI_OBJECT_BED          = 1513,
            AI_OBJECT_CLOSESTOOL   = 1514,
            AI_OBJECT_CUPBOARD     = 1515,    //ng
            AI_OBJECT_REFRIGERATOR = 1516,    //na
            AI_OBJECT_WASHSTAND    = 1517,    //na
            AI_OBJECT_CABINET_TEA_TABLE = 1518,    //na
            AI_OBJECT_CABINET_BED  = 1519,    //na
            AI_OBJECT_CABINET_TV   = 1520,    //na
            AI_OBJECT_DOOR         = 1521,    //ng
            AI_OBJECT_DOOR_THRESHOLD = 1522,    //ng
            AI_OBJECT_WASHING_MACHINE  = 1524,
            
            AI_OBJECT_BED1                  = 1600,
            AI_OBJECT_SOFA1                 = 1601,
            AI_OBJECT_CABINET_BED1          = 1602,
            AI_OBJECT_CUPBOARD1             = 1603,    
            AI_OBJECT_DINING_TABLE1         = 1604,
            AI_OBJECT_REFRIGERATOR1         = 1605,    
            AI_OBJECT_CABINET_TEA_TABLE1    = 1606,    
            AI_OBJECT_CLOSESTOOL1           = 1607,
            AI_OBJECT_CABINET_TV1           = 1608,
            AI_OBJECT_WASHING_MACHINE1      = 1609,
            

            AI_ROOM                = 2000, // room flag
            AI_ROOM_BEDROOM        = 2001,
            AI_ROOM_RESTAURANT     = 2002,
            AI_ROOM_TOILET         = 2003,
            AI_ROOM_CORRIDOR       = 2004,
            AI_ROOM_KITCHEN        = 2005,
            AI_ROOM_LIVING_ROOM    = 2006,
            AI_ROOM_BALCONY        = 2007,
            AI_ROOM_OTHERS         = 2020,

            AI_FLOOR               = 3000, // floor flag
            AI_FLOOR_CONCRETE      = 3001,
            AI_FLOOR_TITLE         = 3002,
            AI_FLOOR_WOOD          = 3003,
            AI_FLOOR_UNKNOW        = 3030,


      
        };

        enum AIFloorBlanketClassLabel
        {
            FloorBlanketClass_blanket       = 0,
            FloorBlanketClass_title_floor   = 1,
            FloorBlanketClass_unknown       = 2,
            FloorBlanketClass_wood_floor    = 3,
            
            SceneClass_drawingRoom          = 4,
            SceneClass_kitchen              = 5,
            SceneClass_bedroom              = 6,
            SceneClass_diningRoom           = 7,
            SceneClass_others               = 8
        };

        //30cls
        enum AIAllClassLabel
        {
            allClass_bed                    = 0,
            allClass_bed1                   = 1,
            allClass_carpet                 = 2,
            allClass_cabinet                = 3,
            // allClass_cat                    = 4,
            allClass_chair_base             = 4,
            allClass_charger                = 5,
            allClass_cupboard               = 6,
            allClass_cupboard1              = 7,
            allClass_dining_table           = 8,
            allClass_dirt                   = 9,
            allClass_door                   = 10,
            // allClass_door_sill              = 13,
            allClass_invalid                = 11,
            allClass_metal_chair_foot       = 12,
            allClass_dog                    = 13,
            // allClass_person                 = 16,
            allClass_pet_feces              = 14,
            allClass_refrigerator           = 15,
            allClass_refrigerator1          = 16,
            allClass_shoe                   = 17,
            allClass_socks                  = 18,
            allClass_sofa                   = 19,
            allClass_sofa1                  = 20,
            // allClass_sweeping_machine       = 24,
            allClass_tea_table              = 21,
            allClass_tile_floor             = 22,
            allClass_toilet                 = 23,
            allClass_TV_stand               = 24,
            allClass_unknown                = 25,
            allClass_washing_machine        = 26,
            allClass_washing_machine1       = 27,
            allClass_weight_scale           = 28,
            allClass_wire                   = 29,
            allClass_wood_floor             = 30,
        };

         enum AIDirtyClassLabeL
         {
              AIDirtyClassLabeL_Water = 0,
              AIDirtyClassLabeL_Powder = 1,
              AIDirtyClassLabeL_Hair = 2,
         };

         //34cls
        // enum AIAllClassLabel
        // {
        //     allClass_bed                    = 0,
        //     allClass_bed1                   = 1,
        //     allClass_carpet                 = 2,
        //     allClass_cabinet                = 3,
        //     // allClass_cat                    = 4,
        //     allClass_chair_base             = 4,
        //     allClass_charger                = 5,
        //     allClass_cupboard               = 6,
        //     allClass_cupboard1              = 7,
        //     allClass_dining_table           = 8,
        //     allClass_dirt                   = 9,
        //     allClass_door                   = 10,
        //     // allClass_door_sill              = 13,
        //     allClass_invalid                = 11,
        //     allClass_metal_chair_foot       = 12,
        //     allClass_person_leg             = 13,
        //     allClass_dog                    = 14,
        //     // allClass_person                 = 16,
        //     allClass_pet_feces              = 15,

        //     allClass_plastic_toy            = 16, //玩具

        //     allClass_refrigerator           = 17,
        //     allClass_refrigerator1          = 18,
        //     allClass_shoe                   = 19,
        //     allClass_socks                  = 20,
        //     allClass_sofa                   = 21,
        //     allClass_sofa1                  = 22,
        //     // allClass_sweeping_machine       = 24,
        //     allClass_tea_table              = 23,
        //     allClass_tile_floor             = 24,
        //     allClass_toilet                 = 25,
        //     allClass_TV_stand               = 26,
        //     allClass_unknown                = 27,
            
        //     allClass_wallet                 = 28, // 钱包,

        //     allClass_washing_machine        = 29,
        //     allClass_washing_machine1       = 30,
        //     allClass_weight_scale           = 31,
        //     allClass_wire                   = 32,
        //     allClass_wood_floor             = 33,
        // };

        struct TAIObejectDetectData 
        {
            TAIObejectDetectData()
            {
                detect_x1 = 0;
                detect_y1 = 0;
                detect_x2 = 0;
                detect_y2 = 0;
                tof_rgb_x1 = 0;
                tof_rgb_y1 = 0;
                tof_rgb_x2 = 0;
                tof_rgb_y2 = 0;
                obj_class = AI_OBJECT_NOTHING;
                obj_ori_class = AI_OBJECT_NOTHING;
                ori_all_class = allClass_unknown;
                ori_floor_class = FloorBlanketClass_unknown;
                object_detect_score = 0.0;
                classify_score = 0.0;
                timestamp = 0;
                detect_img = cv::Mat();
                detect_ori_image_debug_str = "";
            }

            int detect_x1;
            int detect_y1;
            int detect_x2;
            int detect_y2;
            int tof_rgb_x1;
            int tof_rgb_y1;
            int tof_rgb_x2;
            int tof_rgb_y2;
            double object_detect_score;
            double classify_score;
            TAIObjectClass obj_class;
            TAIObjectClass dirty_class;
            TAIObjectClass obj_ori_class;
            AIAllClassLabel ori_all_class;
            AIFloorBlanketClassLabel ori_floor_class;
            long long timestamp;            
            cv::Mat  detect_img;
            std::string detect_ori_image_debug_str;
        };

        struct TCameraCalibresult
        {
            int calib_flag = 0;
            int result_code;
            cv::Mat src_img = cv::Mat();
            cv::Mat res_img = cv::Mat();
            std::string camera_param_str;
        };
            
        //25cls
        // enum AIAllClassLabel
        // {
        //     allClass_bed                    = 0,
        //     allClass_bed1                   = 1,
        //     allClass_carpet                 = 2,
        //     allClass_cabinet                = 3,
        //     allClass_cat                    = 4,
        //     allClass_chair_base             = 5,
        //     allClass_charger                = 6,
        //     allClass_cupboard               = 7,
        //     allClass_cupboard1              = 8,
        //     allClass_dining_table           = 9,
        //     allClass_dirt                   = 10,
        //     allClass_door                   = 12,
        //     allClass_dog                    = 11,
        //     allClass_door_sill              = 13,
        //     allClass_invalid                = 14,
        //     allClass_metal_chair_foot       = 15,
        //     allClass_person                 = 16,
        //     allClass_pet_feces              = 17,
        //     allClass_refrigerator           = 18,
        //     allClass_refrigerator1          = 19,
        //     allClass_shoe                   = 20,
        //     allClass_socks                  = 21,
        //     allClass_sofa                   = 22,
        //     allClass_sofa1                  = 23,
        //     allClass_sweeping_machine       = 24,
        //     allClass_tea_table              = 25,
        //     allClass_tile_floor             = 26,
        //     allClass_toilet                 = 27,
        //     allClass_TV_stand               = 28,
        //     allClass_unknown                = 29,
        //     allClass_washing_machine        = 30,
        //     allClass_washing_machine1       = 31,
        //     allClass_weight_scale           = 32,
        //     allClass_wire                   = 33,
        //     allClass_wood_floor             = 34,
        // };

      
        
    }
}

struct stu{
    stu(){
        a = 10;
    }
    int a = 1;
    int b = 2;

};

#endif
