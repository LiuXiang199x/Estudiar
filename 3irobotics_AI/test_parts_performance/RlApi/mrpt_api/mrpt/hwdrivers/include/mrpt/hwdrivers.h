/* +---------------------------------------------------------------------------+
   |                     Mobile Robot Programming Toolkit (MRPT)               |
   |                          http://www.mrpt.org/                             |
   |                                                                           |
   | Copyright (c) 2005-2014, Individual contributors, see AUTHORS file        |
   | See: http://www.mrpt.org/Authors - All rights reserved.                   |
   | Released under BSD License. See details in http://www.mrpt.org/License    |
   +---------------------------------------------------------------------------+ */

/**  This is the main "include file" for classes into the mrpt::hwdrivers namespace. This file
 *	   includes all the other ones, so user applications must include just this one
 *     and link against the library file "lib_hwdrivers.lib" / "lib_hwdrivers.a"
 */
#ifndef HWDRIVERS_H
#define HWDRIVERS_H

#ifndef MRPT_NO_WARN_BIG_HDR
#include <mrpt/utils/core_defs.h>
MRPT_WARNING("Including <mrpt/hwdrivers.h> makes compilation much slower, consider including only what you need (define MRPT_NO_WARN_BIG_HDR to disable this warning)")
#endif

// Classes into HWDRIVERS
// --------------------------------------------
#include <mrpt/hwdrivers/CGenericSensor.h>
#include <mrpt/hwdrivers/C2DRangeFinderAbstract.h>

#include <mrpt/hwdrivers/CSerialPort.h>
#include <mrpt/hwdrivers/CActivMediaRobotBase.h>
#include <mrpt/hwdrivers/CRoboPeakLidar.h>

#endif
