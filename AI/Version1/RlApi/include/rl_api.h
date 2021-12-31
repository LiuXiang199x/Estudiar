#pragma once
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <fstream>
#include <iostream>
#include <sys/time.h>
#include <vector>
#include <string.h>
#include <uchar.h>
#include <COccupancyGridMap2D.h>

#include "rknn_api.h"


namespace everest
{
	namespace planner
	{
		class RlApi{
			bool processTarget(const int &idx, const int &idy, int &res_idx, int &res_idy);
		};
	}
}
