/* +---------------------------------------------------------------------------+
   |                     Mobile Robot Programming Toolkit (MRPT)               |
   |                          http://www.mrpt.org/                             |
   |                                                                           |
   | Copyright (c) 2005-2014, Individual contributors, see AUTHORS file        |
   | See: http://www.mrpt.org/Authors - All rights reserved.                   |
   | Released under BSD License. See details in http://www.mrpt.org/License    |
   +---------------------------------------------------------------------------+ */

#include "hwdrivers-precomp.h"   // Precompiled headers

#define MRPT_NO_WARN_BIG_HDR
#include <mrpt/hwdrivers.h>

#include <mrpt/utils/CStartUpClassesRegister.h>

using namespace mrpt::utils;
using namespace mrpt::hwdrivers;


void registerAllClasses_mrpt_hwdrivers();

CStartUpClassesRegister  mrpt_hwdrivers_class_reg(&registerAllClasses_mrpt_hwdrivers);


/** Register existing sensors.
  * \sa mrpt::hwdrivers::CGenericSensor::createSensor
  */
void registerAllClasses_mrpt_hwdrivers()
{
	CActivMediaRobotBase::doRegister();
	CRoboPeakLidar::doRegister();
}

