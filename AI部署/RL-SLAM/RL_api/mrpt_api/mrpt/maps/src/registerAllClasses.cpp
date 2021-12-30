/* +---------------------------------------------------------------------------+
   |                     Mobile Robot Programming Toolkit (MRPT)               |
   |                          http://www.mrpt.org/                             |
   |                                                                           |
   | Copyright (c) 2005-2014, Individual contributors, see AUTHORS file        |
   | See: http://www.mrpt.org/Authors - All rights reserved.                   |
   | Released under BSD License. See details in http://www.mrpt.org/License    |
   +---------------------------------------------------------------------------+ */

#include "maps-precomp.h"

#define MRPT_NO_WARN_BIG_HDR
#include <mrpt/maps.h>
#include <mrpt/utils/CStartUpClassesRegister.h>

using namespace mrpt::utils;
using namespace mrpt::slam;

void registerAllClasses_mrpt_maps();

CStartUpClassesRegister  mrpt_maps_class_reg(&registerAllClasses_mrpt_maps);

/*---------------------------------------------------------------
					registerAllClasses_mrpt_maps
  ---------------------------------------------------------------*/
void registerAllClasses_mrpt_maps()
{
	registerClass( CLASS_ID( CPointsMap ) );
	registerClass( CLASS_ID( CSimplePointsMap ) );
	registerClass( CLASS_ID( COccupancyGridMap2D ) );
}

