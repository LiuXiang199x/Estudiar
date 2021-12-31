/* +---------------------------------------------------------------------------+
   |                     Mobile Robot Programming Toolkit (MRPT)               |
   |                          http://www.mrpt.org/                             |
   |                                                                           |
   | Copyright (c) 2005-2014, Individual contributors, see AUTHORS file        |
   | See: http://www.mrpt.org/Authors - All rights reserved.                   |
   | Released under BSD License. See details in http://www.mrpt.org/License    |
   +---------------------------------------------------------------------------+ */

#include "../maps-precomp.h" // Precomp header

#include <mrpt/slam/COccupancyGridMap2D.h>
#include <mrpt/slam/CObservation2DRangeScan.h>
#include <mrpt/slam/CObservationRange.h>
#include <mrpt/utils/round.h> // round()

#include <mrpt/random.h>

using namespace mrpt;
using namespace mrpt::slam;
using namespace mrpt::utils;
using namespace mrpt::random;
using namespace mrpt::poses;
using namespace std;

/*---------------------------------------------------------------
						laserScanSimulator

 Simulates a range scan into the current grid map.
   The simulated scan is stored in a CObservation2DRangeScan object, which is also used
    to pass some parameters: all previously stored characteristics (as aperture,...) are
	  taken into account for simulation. Only a few more parameters are needed. Additive gaussian noise can be optionally added to the simulated scan.
		inout_Scan [IN/OUT] This must be filled with desired parameters before calling, and will contain the scan samples on return.
		robotPose [IN] The robot pose in this map coordinates. Recall that sensor pose relative to this robot pose must be specified in the observation object.
		threshold [IN] The minimum occupancy threshold to consider a cell to be occupied, for example 0.5.
		N [IN] The count of range scan "rays", by default to 361.
		noiseStd [IN] The standard deviation of measurement noise. If not desired, set to 0.
  ---------------------------------------------------------------*/
void  COccupancyGridMap2D::laserScanSimulatorOpt(
		CObservation2DRangeScan	        &inout_Scan,
		const CPose2D					&robotPose,
		float						    threshold,
		size_t							N,
		float						    noiseStd,
		unsigned int				    decimation,
		float							angleNoiseStd ) const
{
	MRPT_START

	ASSERT_(decimation>=1);

	// Sensor pose in global coordinates
	CPose3D		sensorPose3D = CPose3D(robotPose) + inout_Scan.sensorPose;
	// Aproximation: grid is 2D !!!
	CPose2D		sensorPose(sensorPose3D);

	int  initial_idx = x2idx(sensorPose.m_coords[0]);
	int  initial_idy = y2idx(sensorPose.m_coords[1]);
	int  final_idx;
	int  final_idy;
    if(sparialSearch(initial_idx, initial_idy, final_idx, final_idy))
    {
        sensorPose.m_coords[0] = idx2x(final_idx);
        sensorPose.m_coords[1] = idx2y(final_idy);
    }

    // Scan size:
    inout_Scan.scan.resize(N);
    inout_Scan.validRange.resize(N);

    double  A, AA;
	if (inout_Scan.rightToLeft)
	{
		A = sensorPose.phi() - 0.5*inout_Scan.aperture;
		AA = (inout_Scan.aperture / N);
	}
	else
	{
		A = sensorPose.phi() + 0.5*inout_Scan.aperture;
		AA = -(inout_Scan.aperture / N);
	}

    const float free_thres = 1.0f - threshold;
    const unsigned int max_ray_len = round(inout_Scan.maxRange / resolution);

    for (size_t i=0;i<N;i+=decimation,A+=AA*decimation)
    {
    	bool valid;
    	simulateScanRay(
			sensorPose.x(),sensorPose.y(),A,
			inout_Scan.scan[i],valid,
			max_ray_len, free_thres,
			noiseStd, angleNoiseStd );
		inout_Scan.validRange[i] = valid ? 1:0;
    }

	MRPT_END
}

bool  COccupancyGridMap2D::sparialSearch(int &initial_idx,  int &initial_idy,
                                         int &final_idx,    int &final_idy) const
{
    int initial_search_radius = 1;

    int  left_idy  = 0;
    int  up_idx    = 0;
    int  right_idy = 0;
    int  down_idx  = 0;

    bool left_search_out_bound = false;
    bool up_search_out_bound = false;
    bool right_search_out_bound = false;
    bool down_search_out_bound = false;

    while(!(left_search_out_bound && up_search_out_bound && right_search_out_bound && down_search_out_bound))
    {
        /* Left search */
        if(left_search_out_bound)
        {
            left_idy = initial_idy - initial_search_radius - 1;
        }
        else
        {
            int left_out_bound_count = 0;

            for(left_idy = initial_idy - 1; left_idy >= initial_idy - initial_search_radius; left_idy--)
            {
                if(initial_idx < 0 || left_idy < 0 || initial_idx >= static_cast<int>(size_x) || left_idy >= static_cast<int>(size_y))
                {
                    left_out_bound_count++;
                    continue;
                }

                if(getCell(initial_idx, left_idy) > 0.5)
                {
                    final_idx = initial_idx;
                    final_idy = left_idy;
                    return true;
                }
            }

            if(left_out_bound_count == initial_search_radius)
            {
                left_search_out_bound = true;
            }
        }

        /* Up search */
        initial_idy = left_idy + 1;

        if(up_search_out_bound)
        {
            up_idx = initial_idx - initial_search_radius - 1;
        }
        else
        {
            int up_out_bound_count = 0;

            for(up_idx = initial_idx - 1; up_idx >= initial_idx - initial_search_radius; up_idx--)
            {
                if(up_idx < 0 || initial_idy < 0 || up_idx >= static_cast<int>(size_x) || initial_idy >= static_cast<int>(size_y))

                {
                    up_out_bound_count++;
                    continue;
                }

                if(getCell(up_idx, initial_idy) > 0.5)
                {
                    final_idx = up_idx;
                    final_idy = initial_idy;
                    return true;
                }
            }

            if(up_out_bound_count == initial_search_radius)
            {
                up_search_out_bound = true;
            }
        }

        /* Right search */
        initial_search_radius++;
        initial_idx = up_idx + 1;

        if(right_search_out_bound)
        {
            right_idy = initial_idy + initial_search_radius + 1;
        }
        else
        {
            int right_out_bound_count = 0;

            for(right_idy = initial_idy + 1; right_idy <= initial_idy + initial_search_radius; right_idy++)
            {
                if(initial_idx < 0 || right_idy < 0 || initial_idx >= static_cast<int>(size_x) || right_idy >= static_cast<int>(size_y))
                {
                    right_out_bound_count++;
                    continue;
                }

                if(getCell(initial_idx, right_idy) > 0.5)
                {
                    final_idx = initial_idx;
                    final_idy = right_idy;
                    return true;
                }
            }

            if(right_out_bound_count == initial_search_radius)
            {
                right_search_out_bound = true;
            }
        }

        /* Down search */
        initial_idy = right_idy - 1;

        if(down_search_out_bound)
        {
            down_idx = initial_idx + initial_search_radius + 1;
        }
        else
        {
            int down_out_bound_count = 0;

            for(down_idx = initial_idx + 1; down_idx <= initial_idx + initial_search_radius; down_idx++)
            {
                if(down_idx < 0 || initial_idy < 0 || down_idx >= static_cast<int>(size_x) || initial_idy >= static_cast<int>(size_y))
                {
                    down_out_bound_count++;
                    continue;
                }

                if(getCell(down_idx, initial_idy) > 0.5)
                {
                    final_idx = down_idx;
                    final_idy = initial_idy;
                    return true;
                }
            }

            if(down_out_bound_count == initial_search_radius)
            {
                down_search_out_bound = true;
            }
        }

        initial_idx = down_idx - 1;
        initial_search_radius++;
    }
    std::cout << "sparial search fail" << std::endl;
    return false;
}

void  COccupancyGridMap2D::laserScanSimulator(
		CObservation2DRangeScan	        &inout_Scan,
		const CPose2D					&robotPose,
		float						    threshold,
		size_t							N,
		float						    noiseStd,
		unsigned int				    decimation,
		float							angleNoiseStd ) const
{
	MRPT_START

	ASSERT_(decimation>=1);

	// Sensor pose in global coordinates
	CPose3D		sensorPose3D = CPose3D(robotPose) + inout_Scan.sensorPose;
	// Aproximation: grid is 2D !!!
	CPose2D		sensorPose(sensorPose3D);

    // Scan size:
    inout_Scan.scan.resize(N);
    inout_Scan.validRange.resize(N);

    double  A, AA;
	if (inout_Scan.rightToLeft)
	{
		A = sensorPose.phi() - 0.5*inout_Scan.aperture;
		AA = (inout_Scan.aperture / N);
	}
	else
	{
		A = sensorPose.phi() + 0.5*inout_Scan.aperture;
		AA = -(inout_Scan.aperture / N);
	}

    const float free_thres = 1.0f - threshold;
    const unsigned int max_ray_len = round(inout_Scan.maxRange / resolution);

    for (size_t i=0;i<N;i+=decimation,A+=AA*decimation)
    {
    	bool valid;
    	simulateScanRay(
			sensorPose.x(),sensorPose.y(),A,
			inout_Scan.scan[i],valid,
			max_ray_len, free_thres,
			noiseStd, angleNoiseStd );
		inout_Scan.validRange[i] = valid ? 1:0;
    }

	MRPT_END
}

void  COccupancyGridMap2D::sonarSimulator(
		CObservationRange	        &inout_observation,
		const CPose2D				&robotPose,
		float						threshold,
		float						rangeNoiseStd,
		float						angleNoiseStd) const
{
    const float free_thres = 1.0f - threshold;
    const unsigned int max_ray_len = round(inout_observation.maxSensorDistance / resolution);

	for (CObservationRange::iterator itR=inout_observation.begin();itR!=inout_observation.end();++itR)
	{
		const CPose2D sensorAbsolutePose = CPose2D( CPose3D(robotPose) + CPose3D(itR->sensorPose) );

    	// For each sonar cone, simulate several rays and keep the shortest distance:
    	ASSERT_(inout_observation.sensorConeApperture>0)
    	size_t nRays = round(1+ inout_observation.sensorConeApperture / DEG2RAD(1.0) );

    	double direction = sensorAbsolutePose.phi() - 0.5*inout_observation.sensorConeApperture;
		const double Adir = inout_observation.sensorConeApperture / nRays;

		float min_detected_obs=0;
    	for (size_t i=0;i<nRays;i++, direction+=Adir )
    	{
			bool valid;
			float sim_rang;
			simulateScanRay(
				sensorAbsolutePose.x(), sensorAbsolutePose.y(), direction,
				sim_rang, valid,
				max_ray_len, free_thres,
				rangeNoiseStd, angleNoiseStd );

			if (valid && (sim_rang<min_detected_obs || !i))
				min_detected_obs = sim_rang;
    	}
    	// Save:
    	itR->sensedDistance = min_detected_obs;
	}
}

inline void COccupancyGridMap2D::simulateScanRay(
	const double start_x,const double start_y,const double angle_direction,
	float &out_range,bool &out_valid,
	const unsigned int max_ray_len,
	const float threshold_free,
	const double noiseStd, const double angleNoiseStd ) const
{
	const double A_ = angle_direction + randomGenerator.drawGaussian1D_normalized()*angleNoiseStd;

	// Unit vector in the directorion of the ray:
#ifdef HAVE_SINCOS
	double Arx,Ary;
	::sincos(A_, &Ary,&Arx);
	Arx*=resolution;
	Ary*=resolution;
#else
	const double Arx =  cos(A_)*resolution;
	const double Ary =  sin(A_)*resolution;
#endif

	// Ray tracing, until collision, out of the map or out of range:
	unsigned int ray_len=0;
	unsigned int firstUnknownCellDist=max_ray_len+1;
	double rx=start_x;
	double ry=start_y;
	float hitCellOcc = 0.5f;
	int x, y=y2idx(ry);

	while ( (x=x2idx(rx))>=0 && (y=y2idx(ry))>=0 &&
			 x<static_cast<int>(size_x) && y<static_cast<int>(size_y) && (hitCellOcc=getCell(x,y))>threshold_free &&
			 ray_len<max_ray_len  )
	{
		if ( fabs(hitCellOcc-0.5)<0.01f )
			mrpt::utils::keep_min(firstUnknownCellDist, ray_len );

		rx+=Arx;
		ry+=Ary;
		ray_len++;
	}

	// Store:
	// Check out of the grid?
	// Tip: if x<0, (unsigned)(x) will also be >>> size_x ;-)
	if (fabs(hitCellOcc-0.5)<0.01f || static_cast<unsigned>(x)>=size_x || static_cast<unsigned>(y)>=size_y )
	{
		out_valid = false;

		if (firstUnknownCellDist<ray_len)
				out_range = firstUnknownCellDist*resolution;
		else	out_range = ray_len*resolution;
	}
	else
	{ 	// No: The normal case:
		out_range = ray_len*resolution;

        /* Search the neighboor to judge noise point */
		if(IsNeighborFree(x, y, 2 * resolution))
		{
		    out_valid = ray_len<max_ray_len;
		}
		else
		{
		    out_valid = false;
		}

		// Add additive Gaussian noise:
		if (noiseStd>0 && out_valid)
			out_range+=  noiseStd*randomGenerator.drawGaussian1D_normalized();
	}
}

bool  mrpt::slam::COccupancyGridMap2D::IsNeighborFree(const int current_idx,
                                                      const int current_idy,
                                                      const double distance) const
{
    int len = (int)(distance / resolution);
    int counter = 0, threshold = (int)((float)(square(len + len + 1)) * 0.5);///len + len + 1

    for(int idy = current_idy - len; idy <= current_idy + len; idy++)
    {
        if(idy >= (int)size_y || idy < 0)
        {
            continue;
        }

        for(int idx = current_idx - len; idx <= current_idx + len; idx++)
        {
            if(idx >= (int)size_x || idx < 0)
            {
                continue;
            }

            if(getCell(idx, idy) > 0.6 || getCell(idx, idy) < 0.4)
            {
                if(++counter > threshold)
                {
                    return true;
                }
            }
        }
    }

    return false;
}
