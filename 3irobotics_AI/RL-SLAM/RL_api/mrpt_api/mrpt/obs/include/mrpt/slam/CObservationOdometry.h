/* +---------------------------------------------------------------------------+
   |                     Mobile Robot Programming Toolkit (MRPT)               |
   |                          http://www.mrpt.org/                             |
   |                                                                           |
   | Copyright (c) 2005-2014, Individual contributors, see AUTHORS file        |
   | See: http://www.mrpt.org/Authors - All rights reserved.                   |
   | Released under BSD License. See details in http://www.mrpt.org/License    |
   +---------------------------------------------------------------------------+ */

#ifndef CObservationOdometry_H
#define CObservationOdometry_H

#include <mrpt/utils/CSerializable.h>
#include <mrpt/slam/CObservation.h>
#include <mrpt/poses/CPose2D.h>
#include <mrpt/poses/CPose3D.h>

namespace mrpt
{
	namespace slam
	{
		DEFINE_SERIALIZABLE_PRE_CUSTOM_BASE_LINKAGE( CObservationOdometry, CObservation,OBS_IMPEXP  )

		/** An observation of the current (cumulative) odometry for a wheeled robot.
		 *   This kind of observation will only occur in a "observation-only" rawlog file, otherwise
		 *    odometry are modeled with actions. Refer to the <a href="http://www.mrpt.org/Rawlog_Format">page on rawlogs</a>.
		 *
		 * \sa CObservation, CActionRobotMovement2D
		 * \ingroup mrpt_obs_grp
		 */
		class OBS_IMPEXP CObservationOdometry : public CObservation
		{
			// This must be added to any CSerializable derived class:
			DEFINE_SERIALIZABLE( CObservationOdometry )

		 public:
			/** Constructor
			 */
			CObservationOdometry( );

			poses::CPose2D		odometry;		//!< The absolute odometry measurement (IT IS NOT INCREMENTAL)

			bool  hasEncodersInfo; //!< "true" means that "encoderLeftTicks" and "encoderRightTicks" contain valid values.
			int32_t   m_odo_state; // 0:use_pose; 1:use_phi; 2:use_xy

			/** For odometry only: the ticks count for each wheel in ABSOLUTE VALUE (IT IS NOT INCREMENTAL) (positive means FORWARD, for both wheels);
			  * \sa hasEncodersInfo
			  */
			int32_t	 encoderLeftTicks,encoderRightTicks;

			bool	hasVelocities;		//!< "true" means that "velocityLin" and "velocityAng" contain valid values.

			/** The velocity of the robot, linear in meters/sec and angular in rad/sec.
			  */
			float	velocityLin, velocityAng;


			/** A general method to retrieve the sensor pose on the robot.
			  *  It has no effects in this class
			  * \sa setSensorPose
			  */
			void getSensorPose( CPose3D &out_sensorPose ) const { out_sensorPose=CPose3D(0,0,0); }

			/** A general method to change the sensor pose on the robot.
			  *  It has no effects in this class
			  * \sa getSensorPose
			  */
			void setSensorPose( const CPose3D & ) {  }

		}; // End of class def.

	} // End of namespace
} // End of namespace

#endif
