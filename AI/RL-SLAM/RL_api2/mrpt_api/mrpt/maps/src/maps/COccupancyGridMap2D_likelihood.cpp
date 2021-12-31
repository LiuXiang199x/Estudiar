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
// #include <mrpt/slam/CObservation2DRangeScan.h>
#include <mrpt/slam/CObservationRange.h>
#include <mrpt/slam/CSimplePointsMap.h>
#include <mrpt/utils/CStream.h>
#include <mrpt/synch/CCriticalSection.h>


using namespace mrpt;
using namespace mrpt::slam;
using namespace mrpt::utils;
using namespace mrpt::poses;
using namespace std;



/*---------------------------------------------------------------
 Computes the likelihood that a given observation was taken from a given pose in the world being modeled with this map.
	takenFrom The robot's pose the observation is supposed to be taken from.
	obs The observation.
 This method returns a likelihood in the range [0,1].
 ---------------------------------------------------------------*/
double	 COccupancyGridMap2D::computeObservationLikelihood(
			const CObservation		*obs,
			const CPose3D			&takenFrom3D )
{
	CPose2D   takenFrom = CPose2D(takenFrom3D);  // 3D -> 2D, we are in a gridmap...

	switch (likelihoodOptions.likelihoodMethod)
	{
        default:
        case lmRayTracing:
            return computeObservationLikelihood_rayTracing(obs,takenFrom);

        case lmMeanInformation:
            return computeObservationLikelihood_MI(obs,takenFrom);

        case lmConsensus:
            return computeObservationLikelihood_Consensus(obs,takenFrom);

        case lmCellsDifference:
            return computeObservationLikelihood_CellsDifference(obs,takenFrom);

        case lmLikelihoodField_Thrun:
            return computeObservationLikelihood_likelihoodField_Thrun(obs,takenFrom);

        case lmLikelihoodField_II:
            return computeObservationLikelihood_likelihoodField_II(obs,takenFrom);

        case lmConsensusOWA:
            return computeObservationLikelihood_ConsensusOWA(obs,takenFrom);
	};

}

/*---------------------------------------------------------------
 Computes the likelihood that a given observation was taken from a given pose in the world being modeled with this map.
	takenFrom The robot's pose the observation is supposed to be taken from.
	obs The observation.
 This method returns a likelihood in the range [0,1].
 ---------------------------------------------------------------*/
double	 COccupancyGridMap2D::computeObservationLikelihood(
			const CObservation		*obs,
			const CPose2D			&takenFrom,
			double                  &min_logw)
{
	switch (likelihoodOptions.likelihoodMethod)
	{
        default:
        case lmCellsDifference:
            return computeObservationLikelihood_CellsDifference(obs,takenFrom);

        case lmLikelihoodField_Thrun:
            return computeObservationLikelihood_likelihoodField_Thrun(obs, takenFrom, min_logw);

        case lmConsensus:
            return computeObservationLikelihood_likelihoodField_Consensus(obs, takenFrom, min_logw);
	};

}

/*---------------------------------------------------------------
			computeObservationLikelihood_Consensus
---------------------------------------------------------------*/
double	 COccupancyGridMap2D::computeObservationLikelihood_Consensus(
			const CObservation		*obs,
			const CPose2D				&takenFrom )
{
	double		likResult = 0;

	// This function depends on the observation type:
	// -----------------------------------------------------
	if ( obs->GetRuntimeClass() != CLASS_ID(CObservation2DRangeScan) )
	{
		//THROW_EXCEPTION("This method is defined for 'CObservation2DRangeScan' classes only.");
		return 1e-3;
	}
	// Observation is a laser range scan:
	// -------------------------------------------
	const CObservation2DRangeScan		*o = static_cast<const CObservation2DRangeScan*>( obs );

	// Insert only HORIZONTAL scans, since the grid is supposed to
	//  be a horizontal representation of space.
	if ( ! o->isPlanarScan(insertionOptions.horizontalTolerance) ) return 0.5f;		// NO WAY TO ESTIMATE NON HORIZONTAL SCANS!!

	// Assure we have a 2D points-map representation of the points from the scan:
	const CPointsMap *compareMap = o->buildAuxPointsMap<mrpt::slam::CPointsMap>();

	// Observation is a points map:
	// -------------------------------------------
	size_t			Denom=0;
//	int			Acells = 1;
	TPoint2D pointGlobal,pointLocal;


	// Get the points buffers:

	//	compareMap.getPointsBuffer( n, xs, ys, zs );
	const size_t n = compareMap->size();

	for (size_t i=0;i<n;i+=likelihoodOptions.consensus_takeEachRange)
	{
		// Get the point and pass it to global coordinates:
		compareMap->getPoint(i,pointLocal);
		takenFrom.composePoint(pointLocal, pointGlobal);

		int		cx0 = x2idx( pointGlobal.x );
		int		cy0 = y2idx( pointGlobal.y );

		likResult += 1-getCell_nocheck(cx0,cy0);
		Denom++;
	}
	if (Denom)	likResult/=Denom;
	likResult = pow(likResult, static_cast<double>( likelihoodOptions.consensus_pow ) );

	return log(likResult);
}

/*---------------------------------------------------------------
			computeObservationLikelihood_ConsensusOWA
---------------------------------------------------------------*/
double	 COccupancyGridMap2D::computeObservationLikelihood_ConsensusOWA(
			const CObservation		*obs,
			const CPose2D				&takenFrom )
{
	double		likResult = 0;

	// This function depends on the observation type:
	// -----------------------------------------------------
	if ( obs->GetRuntimeClass() == CLASS_ID(CObservation2DRangeScan) )
	{
		//THROW_EXCEPTION("This method is defined for 'CObservation2DRangeScan' classes only.");
		return 1e-3;
	}
	// Observation is a laser range scan:
	// -------------------------------------------
	const CObservation2DRangeScan		*o = static_cast<const CObservation2DRangeScan*>( obs );

	// Insert only HORIZONTAL scans, since the grid is supposed to
	//  be a horizontal representation of space.
	if ( ! o->isPlanarScan(insertionOptions.horizontalTolerance) ) return 0.5;		// NO WAY TO ESTIMATE NON HORIZONTAL SCANS!!

	// Assure we have a 2D points-map representation of the points from the scan:
	CPointsMap::TInsertionOptions	insOpt;
	insOpt.minDistBetweenLaserPoints	= -1;		// ALL the laser points

	const CPointsMap *compareMap = o->buildAuxPointsMap<mrpt::slam::CPointsMap>( &insOpt );

	// Observation is a points map:
	// -------------------------------------------
	int				Acells = 1;
	TPoint2D		pointGlobal,pointLocal;

	// Get the points buffers:
	const size_t n = compareMap->size();

	// Store the likelihood values in this vector:
	likelihoodOutputs.OWA_pairList.clear();
	for (size_t i=0;i<n;i++)
	{
		// Get the point and pass it to global coordinates:
		compareMap->getPoint(i,pointLocal);
		takenFrom.composePoint(pointLocal, pointGlobal);

		int		cx0 = x2idx( pointGlobal.x );
		int		cy0 = y2idx( pointGlobal.y );

		int		cxMin = max(0,cx0 - Acells);
		int		cxMax = min(static_cast<int>(size_x)-1,cx0 + Acells);
		int		cyMin = max(0,cy0 - Acells);
		int		cyMax = min(static_cast<int>(size_y)-1,cy0 + Acells);

		double	lik = 0;

		for (int cx=cxMin;cx<=cxMax;cx++)
			for (int cy=cyMin;cy<=cyMax;cy++)
				lik += 1-getCell_nocheck(cx,cy);

		int		nCells = (cxMax-cxMin+1)*(cyMax-cyMin+1);
		ASSERT_(nCells>0);
		lik/=nCells;

		TPairLikelihoodIndex	element;
		element.first = lik;
		element.second = pointGlobal;
		likelihoodOutputs.OWA_pairList.push_back( element );
	} // for each range point

	// Sort the list of likelihood values, in descending order:
	// ------------------------------------------------------------
	std::sort(likelihoodOutputs.OWA_pairList.begin(),likelihoodOutputs.OWA_pairList.end());

	// Cut the vector to the highest "likelihoodOutputs.OWA_length" elements:
	size_t	M = likelihoodOptions.OWA_weights.size();
	ASSERT_( likelihoodOutputs.OWA_pairList.size()>=M );

	likelihoodOutputs.OWA_pairList.resize(M);
	likelihoodOutputs.OWA_individualLikValues.resize( M );
	likResult = 0;
	for (size_t k=0;k<M;k++)
	{
		likelihoodOutputs.OWA_individualLikValues[k] = likelihoodOutputs.OWA_pairList[k].first;
		likResult+= likelihoodOptions.OWA_weights[k] * likelihoodOutputs.OWA_individualLikValues[k];
	}

	return log(likResult);
}

/*---------------------------------------------------------------
			computeObservationLikelihood_CellsDifference
---------------------------------------------------------------*/
double	 COccupancyGridMap2D::computeObservationLikelihood_CellsDifference(
			const CObservation		*obs,
			const CPose2D				&takenFrom )
{
    double		ret = 1.0;
    double		log_ret = 0;

	if ( obs->GetRuntimeClass() == CLASS_ID(CObservation2DRangeScan) )
    {
		const CObservation2DRangeScan		*o = static_cast<const CObservation2DRangeScan*>( obs );

        ///for once time
        CPointsMap::TInsertionOptions		opts;
		opts.minDistBetweenLaserPoints	= o->min_interval_dis;//resolution*0.5f
		opts.isPlanarMap				= true; // Already filtered above!
		opts.horizontalTolerance		= insertionOptions.horizontalTolerance;

		const CPointsMap *points_map1 = o->buildAuxPointsMap<mrpt::slam::CPointsMap>(&opts);
		///

		int         invalid_size = 0;
		float       sum_weight = 0.0;
        TPoint3D	pointLocal;

		if(points_map1 != nullptr)
        {
            invalid_size = points_map1->size();
            for (size_t j=0; j<invalid_size; j++)
            {
                points_map1->getPoint(j,pointLocal);
                sum_weight = sum_weight + pointLocal.z;
            }
        }
        ///

		COccupancyGridMap2D		compareGrid(takenFrom.x()-8.5,takenFrom.x()+8.5,takenFrom.y()-8.5,takenFrom.y()+8.5,resolution);
		CPose3D					robotPose(takenFrom);
		int						Ax, Ay;

		compareGrid.insertionOptions.maxDistanceInsertion			= 8.0;
		compareGrid.insertionOptions.maxOccupancyUpdateCertainty	= 0.95f;
		o->insertObservationInto( &compareGrid, &robotPose );
		o->insertObservationInto( &compareGrid, &robotPose );

		Ax = round((x_min - compareGrid.x_min) / resolution);
		Ay = round((y_min - compareGrid.y_min) / resolution);

		float		nCellsCompared = 0;
		float		cellsDifference = 0;
		int			x0 = max(0,Ax);
		int			y0 = max(0,Ay);
		int			x1 = min(compareGrid.size_x, size_x+Ax);
		int			y1 = min(compareGrid.size_y, size_y+Ay);

		for (int x=x0;x<x1;x+=1)
		{
			for (int y=y0;y<y1;y+=1)
			{
				float	xx = compareGrid.idx2x(x);
				float	yy = compareGrid.idx2y(y);

				float	c1 = getPos(xx,yy);
				float	c2 = compareGrid.getCell(x,y);
				/*if ( c2<0.45f || c2>0.55f )
				{
					nCellsCompared++;
					if ((c1>0.5 && c2<0.5) || (c1<0.5 && c2>0.5))
						cellsDifference++;
				}*/
				if (c2 > 0.55f)//white?
				{
					nCellsCompared = nCellsCompared + 1;
//					if (c1 < 0.50005f)//black and gray
					if (c1 < 0.4995f)//black
                    {
                        if(0)
                        {
                            cellsDifference = cellsDifference + 1;
                        }
                        else
                        {
                            bool valid_flag = false;

                            int black_x0 = max(0, (x - 1));
                            int black_y0 = max(0, (y - 1));
                            int	black_x1 = min(compareGrid.size_x, uint32_t(x + 1));
                            int	black_y1 = min(compareGrid.size_y, uint32_t(y + 1));
                            for(int temp_x = black_x0; temp_x <= black_x1; temp_x++)
                            {
                                for(int temp_y = black_y0; temp_y <= black_y1; temp_y++)
                                {
//                                    float	temp_xx = compareGrid.idx2x(temp_x);
//                                    float	temp_yy = compareGrid.idx2y(temp_y);
//                                    if(temp_xx > 0 && temp_xx < size_x && temp_yy > 0 && temp_yy < size_y)
                                    {
                                        if(compareGrid.getCell(temp_x,temp_y) < 0.4995f)//black
                                        {
                                            valid_flag = true;
                                            break;
                                        }
                                    }
                                }
                                if(valid_flag)   break;
                            }
                            if(!valid_flag)   cellsDifference = cellsDifference + 1;
                        }
                    }
				}
			}
		}
		if(nCellsCompared > 0)
        {
            ret = 1 - cellsDifference / (nCellsCompared);
        }

        log_ret = log(ret);

        if(invalid_size > 0)
        {
//            log_ret = log_ret * invalid_size;
            log_ret = log_ret * sum_weight;
        }
	 }

	 return log_ret;
}

/*---------------------------------------------------------------
			computeObservationLikelihood_MI
---------------------------------------------------------------*/
double	 COccupancyGridMap2D::computeObservationLikelihood_MI(
			const CObservation		*obs,
			const CPose2D				&takenFrom )
{
	MRPT_START

 	CPose3D			poseRobot(takenFrom);
	double			res;

	// Dont modify the grid, only count the changes in Information
	updateInfoChangeOnly.enabled = true;
	insertionOptions.maxDistanceInsertion*= likelihoodOptions.MI_ratio_max_distance;

	// Reset the new information counters:
	updateInfoChangeOnly.cellsUpdated = 0;
	updateInfoChangeOnly.I_change = 0;
	updateInfoChangeOnly.laserRaysSkip = likelihoodOptions.MI_skip_rays;

	// Insert the observation (It will not be really inserted, only the information counted)
	insertObservation(obs,&poseRobot);

	// Compute the change in I aported by the observation:
	double	newObservation_mean_I;
	if (updateInfoChangeOnly.cellsUpdated)
			newObservation_mean_I = updateInfoChangeOnly.I_change / updateInfoChangeOnly.cellsUpdated;
	else	newObservation_mean_I = 0;

	// Let the normal mode enabled, i.e. the grid can be updated
	updateInfoChangeOnly.enabled = false;
	insertionOptions.maxDistanceInsertion/=likelihoodOptions.MI_ratio_max_distance;


	res = pow(newObservation_mean_I, static_cast<double>(likelihoodOptions.MI_exponent) );

	return log(res);

	MRPT_END
 }

double COccupancyGridMap2D::computeObservationLikelihood_rayTracing(const CObservation *obs, const CPose2D &takenFrom )
{
    double		ret=0;

	if(obs->GetRuntimeClass() == CLASS_ID(CObservation2DRangeScan))
    {
        const CObservation2DRangeScan *o = static_cast<const CObservation2DRangeScan*>( obs );

        CObservation2DRangeScan		simulatedObs;
        simulatedObs.aperture = o->aperture;
        simulatedObs.maxRange = o->maxRange;
        simulatedObs.rightToLeft = o->rightToLeft;
        simulatedObs.sensorPose = o->sensorPose;

        int	decimation = 1;
        int	nRays = o->scan.size();
        laserScanSimulator(simulatedObs, takenFrom, 0.50f, nRays, 0, decimation);

        static mrpt::synch::CCriticalSection    debug_simulate_cs;
        {
            mrpt::synch::CCriticalSectionLocker lock(&debug_simulate_cs);
            m_simulatedObs = simulatedObs;
        }

        int debug_count = 0;
//        for(int m = 0; m < simulatedObs.scan.size(); m++)
//        {
//            if (!o->validRange[m])
//            {
//                simulatedObs.validRange[m] = false;
//                simulatedObs.scan[m] = 0.0;
//            }
//            if(simulatedObs.scan[m] > 8.0)
//            {
//                simulatedObs.scan[m] = 0.0;
//            }
//        }

////        if(1)
////        {
//            CPose3D					robotPose(takenFrom);
//
//            COccupancyGridMap2D		original_Grid(takenFrom.x()-10,takenFrom.x()+10,takenFrom.y()-10,takenFrom.y()+10,resolution);
//            COccupancyGridMap2D		simulate_Grid(takenFrom.x()-10,takenFrom.x()+10,takenFrom.y()-10,takenFrom.y()+10,resolution);
//
//            original_Grid.insertionOptions.maxDistanceInsertion			= 5.0;
//            original_Grid.insertionOptions.maxOccupancyUpdateCertainty	= 0.95f;
//            o->insertObservationInto(&original_Grid, &robotPose );
//            o->insertObservationInto(&original_Grid, &robotPose );
//
//            simulate_Grid.insertionOptions.maxDistanceInsertion			= 5.0;
//            simulate_Grid.insertionOptions.maxOccupancyUpdateCertainty	= 0.95f;
//            simulatedObs.insertObservationInto(&simulate_Grid, &robotPose );
//            simulatedObs.insertObservationInto(&simulate_Grid, &robotPose );
//
//            float	    CellsCompared = 0;
//            float		cellsDifference = 0;
//            float       cell_num1 = 0;
//            float       cell_num2 = 0;
//
//            for (int x = 0; x < simulate_Grid.size_x; x += 1)
//            {
//                for (int y = 0; y < simulate_Grid.size_y; y += 1)
//                {
//                    float	c1 = original_Grid.getCell(x, y);
//                    float	c2 = simulate_Grid.getCell(x, y);
//
//                    if (c1<0.5f || c1>0.5f)
//                    {
//                        cell_num1 = cell_num1 + 1;
//                    }
//                    if (c2<0.5f || c2>0.5f)
//                    {
//                        cell_num2 = cell_num2 + 1;
//                    }
//                    if((c1 < 0.5f && c2 < 0.5f) || (c1 > 0.5f && c2 > 0.5f))
//                    {
//                        cellsDifference = cellsDifference + 1;
//                    }
//                }
//            }
////            CellsCompared = cell_num1 > cell_num2 ? cell_num1 : cell_num2;
//            CellsCompared = (cell_num1 + cell_num2) / 2.0;
////            ret = cellsDifference / (CellsCompared);
////            ret = log(ret);
////        }
//        else
//        {
            double		stdLaser   = 0.05;///
            double		stdSqrt2 = sqrt(2.0f) * stdLaser;
            float		r_sim,r_obs;
            double		likelihood;
            double      valid_count = 0.0;

            for (int j=0;j<nRays;j+=decimation)
            {
                r_sim = simulatedObs.scan[j];
                r_obs = o->scan[j];

                if (o->validRange[j])
                {
                    likelihood = 0.1/o->maxRange + 0.9*exp( -square( min((float)fabs(r_sim-r_obs),2.0f)/stdSqrt2) );
                    ret += log(likelihood);
//                    likelihood = likelihood + exp( -square( min((float)fabs(r_sim-r_obs),2.0f)/stdSqrt2) );
//                    valid_count = valid_count + 1;
                }
            }
//            ret = 0.1 * ret + 0.9 * log(likelihood / valid_count);
//        }
    }

    return ret;
}
/**/

/*---------------------------------------------------------------
			computeObservationLikelihood_likelihoodField_Thrun
---------------------------------------------------------------*/
double	 COccupancyGridMap2D::computeObservationLikelihood_likelihoodField_Thrun(
			const CObservation		*obs,
			const CPose2D				&takenFrom )
{
	MRPT_START
	double		ret=0;
	if ( IS_CLASS(obs, CObservation2DRangeScan) )
	{
		const CObservation2DRangeScan		*o = static_cast<const CObservation2DRangeScan*>( obs );
		if (!o->isPlanarScan(insertionOptions.horizontalTolerance)) return -10;

		CPointsMap::TInsertionOptions		opts;
		opts.minDistBetweenLaserPoints	= o->min_interval_dis;//resolution*0.5f
		opts.isPlanarMap				= true; // Already filtered above!
		opts.horizontalTolerance		= insertionOptions.horizontalTolerance;

		ret = computeLikelihoodField_Thrun( o->buildAuxPointsMap<mrpt::slam::CPointsMap>(&opts), &takenFrom );
	}
	return ret;
	MRPT_END
}

double	 COccupancyGridMap2D::computeObservationLikelihood_likelihoodField_Thrun(
			const CObservation		*obs,
			const CPose2D				&takenFrom,
			double                      &min_logw)
{
	MRPT_START
	double		ret=0;

    const CObservation2DRangeScan		*o = static_cast<const CObservation2DRangeScan*>( obs );

    CPointsMap::TInsertionOptions		opts;
    opts.minDistBetweenLaserPoints	= o->min_interval_dis;//resolution*0.5f
    opts.isPlanarMap				= true; // Already filtered above!
    opts.horizontalTolerance		= insertionOptions.horizontalTolerance;

    ret = computeLikelihoodField_Thrun( o->buildAuxPointsMap<mrpt::slam::CPointsMap>(&opts), min_logw, &takenFrom);

	return ret;

	MRPT_END

}

double	 COccupancyGridMap2D::computeObservationLikelihood_likelihoodField_Consensus(
			const CObservation		*obs,
			const CPose2D				&takenFrom,
			double                      &min_logw)
{
	MRPT_START
	double		ret=0;

    const CObservation2DRangeScan		*o = static_cast<const CObservation2DRangeScan*>( obs );

    CPointsMap::TInsertionOptions		opts;
    opts.minDistBetweenLaserPoints	= o->min_interval_dis;//resolution*0.5f
    opts.isPlanarMap				= true; // Already filtered above!
    opts.horizontalTolerance		= insertionOptions.horizontalTolerance;

    ret = computeLikelihoodField_Consensus( o->buildAuxPointsMap<mrpt::slam::CPointsMap>(&opts), min_logw, &takenFrom);

	return ret;

	MRPT_END

}

/*---------------------------------------------------------------
		computeObservationLikelihood_likelihoodField_II
---------------------------------------------------------------*/
double	 COccupancyGridMap2D::computeObservationLikelihood_likelihoodField_II(
			const CObservation		*obs,
			const CPose2D				&takenFrom )
{
	MRPT_START

	double		ret=0;

	// This function depends on the observation type:
	// -----------------------------------------------------
	if ( obs->GetRuntimeClass() == CLASS_ID(CObservation2DRangeScan) )
	{
		// Observation is a laser range scan:
		// -------------------------------------------
		const CObservation2DRangeScan		*o = static_cast<const CObservation2DRangeScan*>( obs );

		// Insert only HORIZONTAL scans, since the grid is supposed to
		//  be a horizontal representation of space.
		if (!o->isPlanarScan(insertionOptions.horizontalTolerance)) return 0.5f;	// NO WAY TO ESTIMATE NON HORIZONTAL SCANS!!

		// Assure we have a 2D points-map representation of the points from the scan:

		// Compute the likelihood of the points in this grid map:
		ret = computeLikelihoodField_II( o->buildAuxPointsMap<mrpt::slam::CPointsMap>(), &takenFrom );

	} // end of observation is a scan range 2D

	return ret;

	MRPT_END

}


/*---------------------------------------------------------------
					computeLikelihoodField_Thrun
 ---------------------------------------------------------------*/
double	 COccupancyGridMap2D::computeLikelihoodField_Thrun( const CPointsMap	*pm, const CPose2D *relativePose )
{
	MRPT_START

	double		    ret = 0;
	size_t		    N = pm->size();

	if (!N)
	{
		return -100000; // No way to estimate this likelihood!!
	}

	float		    stdHit	= likelihoodOptions.LF_stdHit;
	float		    zHit	= likelihoodOptions.LF_zHit;
	float		    zRandom	= likelihoodOptions.LF_zRandom;
	float		    zRandomMaxRange	= likelihoodOptions.LF_maxRange;
	float		    zRandomTerm = zRandom / zRandomMaxRange;
	float		    Q = -0.5f / square(stdHit);
    double		    maxCorrDist_sq = square(likelihoodOptions.LF_maxCorrsDistance);
	double		    minimumLik = zRandomTerm  + zHit * exp( Q * maxCorrDist_sq );

	unsigned int	size_x_1 = size_x-1;
	unsigned int	size_y_1 = size_y-1;

	double		    thisLik;
	TPoint2D	    pointLocal;
	TPoint2D	    pointGlobal;

    double          particle_pose_ssin;
	double		    particle_pose_ccos;
	double          particle_pose_x;
	double          particle_pose_y;

    if (relativePose)
    {
        ::sincos(relativePose->phi(), &particle_pose_ssin,&particle_pose_ccos);
        particle_pose_x = relativePose->x();
	    particle_pose_y = relativePose->y();
    }
    else
    {
        cout << "ERROR----NO PARTICLE POSE" << endl;
    }

#define LIK_LF_CACHE_INVALID    (66)

	for (size_t j=0;j<N;j++)
	{
		// Get the point and pass it to global coordinates:
        pm->getPoint(j,pointLocal);

        pointGlobal.x = particle_pose_x + pointLocal.x * particle_pose_ccos - pointLocal.y * particle_pose_ssin;
        pointGlobal.y = particle_pose_y + pointLocal.x * particle_pose_ssin + pointLocal.y * particle_pose_ccos;

		// Point to cell indixes
		int cx = x2idx( pointGlobal.x );
		int cy = y2idx( pointGlobal.y );

		// Precomputed table:
		// Tip: Comparison cx<0 is implicit in (unsigned)(x)>size...
		if ( static_cast<unsigned>(cx)>=size_x_1 || static_cast<unsigned>(cy)>=size_y_1 )
		{
			// We are outside of the map: Assign the likelihood for the max. correspondence distance:
			thisLik = minimumLik;
		}
		else
		{
			// We are into the map limits:
            thisLik = precomputedLikelihood[ cx+cy*size_x ];

			if (thisLik==LIK_LF_CACHE_INVALID )
			{
				thisLik = minimumLik;
			}
		}
        ret += log(thisLik);
	}

	return ret;

	MRPT_END
}

double	 COccupancyGridMap2D::computeLikelihoodField_Thrun( const CPointsMap	*pm, double &min_logw, const CPose2D *relativePose)
{
	MRPT_START

	double		    ret = 0;
	size_t		    N = pm->size();

	if (!N)
	{
		return -100000; // No way to estimate this likelihood!!
	}
	double		    minimumLik = 0.658558;

	unsigned int	size_x_1 = size_x-1;
	unsigned int	size_y_1 = size_y-1;

	double		    thisLik;
	TPoint3D	    pointLocal;
	TPoint2D	    pointGlobal;

    double          particle_pose_ssin;
	double		    particle_pose_ccos;
	double          particle_pose_x;
	double          particle_pose_y;

    if (relativePose)
    {
        ::sincos(relativePose->phi(), &particle_pose_ssin,&particle_pose_ccos);
        particle_pose_x = relativePose->x();
	    particle_pose_y = relativePose->y();
    }
    else
    {
        cout << "ERROR----NO PARTICLE POSE" << endl;
    }

#define LIK_LF_CACHE_INVALID    (66)

	for (size_t j=0;j<N;j++)
	{
        pm->getPoint(j,pointLocal);

        pointGlobal.x = particle_pose_x + pointLocal.x * particle_pose_ccos - pointLocal.y * particle_pose_ssin;
        pointGlobal.y = particle_pose_y + pointLocal.x * particle_pose_ssin + pointLocal.y * particle_pose_ccos;

		int cx = x2idx( pointGlobal.x );
		int cy = y2idx( pointGlobal.y );

		if ( static_cast<unsigned>(cx)>=size_x_1 || static_cast<unsigned>(cy)>=size_y_1 )
		{
			thisLik = minimumLik;
		}
		else
		{
            thisLik = precomputedLikelihood[ cx+cy*size_x ];

			if (thisLik==LIK_LF_CACHE_INVALID )
			{
				thisLik = minimumLik;
			}
		}
        ret += pointLocal.z * log(thisLik);

        if(ret < min_logw)
        {
            ret = -1e300;
            break;
        }
	}

	return ret;

	MRPT_END
}

double	 COccupancyGridMap2D::computeLikelihoodField_Consensus( const CPointsMap	*pm, double &min_logw, const CPose2D *relativePose)
{
	MRPT_START

	double		    ret = 0;
	size_t		    N = pm->size();

	if (!N)
	{
		return -100000; // No way to estimate this likelihood!!
	}

	float		    stdHit	= likelihoodOptions.LF_stdHit;
	float		    zHit	= likelihoodOptions.LF_zHit;
	float		    zRandom	= likelihoodOptions.LF_zRandom;
	float		    zRandomMaxRange	= likelihoodOptions.LF_maxRange;
	float		    zRandomTerm = zRandom / zRandomMaxRange;
	float		    Q = -0.5f / square(stdHit);
    double		    maxCorrDist_sq = square(likelihoodOptions.LF_maxCorrsDistance);
	double		    minimumLik = zRandomTerm  + zHit * exp( Q * maxCorrDist_sq );

	unsigned int	size_x_1 = size_x-1;
	unsigned int	size_y_1 = size_y-1;

	double		    thisLik;
//	TPoint2D	    pointLocal;
	TPoint3D	    pointLocal;
	TPoint2D	    pointGlobal;

    double          particle_pose_ssin;
	double		    particle_pose_ccos;
	double          particle_pose_x;
	double          particle_pose_y;

    if (relativePose)
    {
        ::sincos(relativePose->phi(), &particle_pose_ssin,&particle_pose_ccos);
        particle_pose_x = relativePose->x();
	    particle_pose_y = relativePose->y();
    }
    else
    {
        cout << "ERROR----NO PARTICLE POSE" << endl;
    }

#define LIK_LF_CACHE_INVALID    (66)

	for (size_t j=0;j<N;j++)
	{
		// Get the point and pass it to global coordinates:
        pm->getPoint(j,pointLocal);

        pointGlobal.x = particle_pose_x + pointLocal.x * particle_pose_ccos - pointLocal.y * particle_pose_ssin;
        pointGlobal.y = particle_pose_y + pointLocal.x * particle_pose_ssin + pointLocal.y * particle_pose_ccos;

		// Point to cell indixes
		int cx = x2idx( pointGlobal.x );
		int cy = y2idx( pointGlobal.y );

		// Precomputed table:
		// Tip: Comparison cx<0 is implicit in (unsigned)(x)>size...
		if ( static_cast<unsigned>(cx)>=size_x_1 || static_cast<unsigned>(cy)>=size_y_1 )
		{
			// We are outside of the map: Assign the likelihood for the max. correspondence distance:
			thisLik = minimumLik;
		}
		else
		{
			// We are into the map limits:
            thisLik = precomputedLikelihood[ cx+cy*size_x ];

            if(thisLik < 0.35 && thisLik > 0.25)//0.3
            {
                thisLik = 0.5;//0.5
            }
//            else if(thisLik < 0.45 && thisLik > 0.35)//0.4
//            {
//                thisLik = 0.3;
//            }

			if (thisLik==LIK_LF_CACHE_INVALID )
			{
				thisLik = minimumLik;
			}
		}
//        ret += log(thisLik);

        ///use norm weight
        ret += pointLocal.z * log(thisLik);

        if(ret < min_logw)
        {
            ret = -1e300;
            break;
        }

	}

	return ret;

	MRPT_END
}

/*---------------------------------------------------------------
					clearPrecomputedVector
 ---------------------------------------------------------------*/
void COccupancyGridMap2D::clearPrecomputedVector()
{
    precomputedLikelihood.clear();

    std::vector<float> temp1;
	m_precomputed_coarse.swap(temp1);
	m_precomputed_coarse.clear();

	std::vector<float> temp2;
	m_precomputed_fine.swap(temp2);
	m_precomputed_fine.clear();
}

void COccupancyGridMap2D::releasePrecomputeVectorCache()
{
    std::vector<float> temp;
    precomputedLikelihood.swap(temp);

    std::vector<float> temp1;
	m_precomputed_coarse.swap(temp1);
	m_precomputed_coarse.clear();

	std::vector<float> temp2;
	m_precomputed_fine.swap(temp2);
	m_precomputed_fine.clear();
}

void COccupancyGridMap2D::releaseMapCache()
{
    std::vector<cellType> temp;
    map.swap(temp);
}

/*---------------------------------------------------------------
					getPrecomputedVector
 ---------------------------------------------------------------*/
const std::vector<float> &COccupancyGridMap2D::getPrecomputedVector()
{
    return precomputedLikelihood;
}

const std::vector<float> &COccupancyGridMap2D::getPrecomputedCoarseVector()
{
    return m_precomputed_coarse;
}

const std::vector<float> &COccupancyGridMap2D::getPrecomputedFineVector()
{
    return m_precomputed_fine;
}

/*---------------------------------------------------------------
					setPrecomputedVector
 ---------------------------------------------------------------*/
void COccupancyGridMap2D::setPrecomputedVector(const std::vector<float>  &temp_vector)
{
    precomputedLikelihood.assign(temp_vector.begin(), temp_vector.end());
}

void COccupancyGridMap2D::setPrecomputedCoarseVector(const std::vector<float>  &temp_vector)
{
    m_precomputed_coarse.assign(temp_vector.begin(), temp_vector.end());
}

void COccupancyGridMap2D::setPrecomputedFineVector(const std::vector<float>  &temp_vector)
{
    m_precomputed_fine.assign(temp_vector.begin(), temp_vector.end());
}

/*---------------------------------------------------------------
					setPrecomputedVector2
 ---------------------------------------------------------------*/
void COccupancyGridMap2D::setPrecomputedVector2(const std::vector<float>  &temp_vector)
{
    precomputedLikelihood = temp_vector;
}

/*---------------------------------------------------------------
					updatePrecomputedVector2InputVector
 ---------------------------------------------------------------*/
void COccupancyGridMap2D::updatePrecomputedVector2InputVector(std::vector<double>    &input_vector,
                                                              int input_xx_min, int input_xx_max,
                                                              int input_yy_min, int input_yy_max)
{
    int cx = 0;
    int cy = 0;

    for(cx = input_xx_min; cx < input_xx_max; cx++)
    {
        for(cy = input_yy_min; cy < input_yy_max; cy++)
        {
            input_vector[ cx+cy*size_x ] = precomputedLikelihood[ cx+cy*size_x ];
        }
    }
}

/*---------------------------------------------------------------
					precomputedLikelihood_Thrun
 ---------------------------------------------------------------*/
void	 COccupancyGridMap2D::precomputedLikelihood_Thrun(const int            &x_min,
                                                          const int            &y_min,
                                                          const int            &x_max,
                                                          const int            &y_max)
{
	MRPT_START
    likelihoodOptions.LF_maxCorrsDistance = 0.3f;
	int		    K = (int)ceil(likelihoodOptions.LF_maxCorrsDistance/*m*/ / resolution);	// The size of the checking area for matchings:
	bool		Product_T_OrSum_F = !likelihoodOptions.LF_alternateAverageMethod;

//	float		stdHit	= likelihoodOptions.LF_stdHit;
	float		stdHit	= 0.3;
	float		zHit	= likelihoodOptions.LF_zHit;
	float		zRandom	= likelihoodOptions.LF_zRandom;
	float		zRandomMaxRange	= likelihoodOptions.LF_maxRange;
	float		zRandomTerm = zRandom / zRandomMaxRange;
	float		Q = -0.5f / square(stdHit);

	unsigned int	size_x_1 = size_x-1;
	unsigned int	size_y_1 = size_y-1;

	// Aux. variables for the "for j" loop:
	double		thisLik;
	double		maxCorrDist_sq = square(likelihoodOptions.LF_maxCorrsDistance);
	float		occupiedMinDist;

	cellType	thresholdCellValue = p2l(0.5f);

	#define LIK_LF_CACHE_INVALID    (66)

    if (map.size())
            precomputedLikelihood.assign( map.size(),LIK_LF_CACHE_INVALID);
    else	precomputedLikelihood.clear();

	const double _resolution = this->resolution;
	const double constDist2DiscrUnits = 100 / (_resolution * _resolution);
	const double constDist2DiscrUnits_INV = 1.0 / constDist2DiscrUnits;

	int cx;
	int cy;

    for(cx = x_min; cx < x_max; cx++)
    {
        for(cy = y_min; cy < y_max; cy++)
        {
            occupiedMinDist = maxCorrDist_sq; // The max.distance

            // Compute now:
            // -------------
            // Find the closest occupied cell in a certain range, given by K:
            int xx1 = max(0,cx-K);
            int xx2 = min(size_x_1,(unsigned)(cx+K));
            int yy1 = max(0,cy-K);
            int yy2 = min(size_y_1,(unsigned)(cy+K));

            // Optimized code: this part will be invoked a *lot* of times:
            {
                cellType  *mapPtr  = &map[xx1+yy1*size_x]; // Initial pointer position
                unsigned   incrAfterRow = size_x - ((xx2-xx1)+1);

                signed int Ax0 = 10*(xx1-cx);
                signed int Ay  = 10*(yy1-cy);

                unsigned int occupiedMinDistInt = mrpt::utils::round( maxCorrDist_sq * constDist2DiscrUnits );

                for (int yy=yy1;yy<=yy2;yy++)
                {
                    unsigned int Ay2 = square((unsigned int)(Ay)); // Square is faster with unsigned.
                    signed short Ax=Ax0;
                    cellType  cell;

                    for (int xx=xx1;xx<=xx2;xx++)
                    {
                        if ( (cell =*mapPtr++) < thresholdCellValue )
                        {
                            unsigned int d = square((unsigned int)(Ax)) + Ay2;
                            keep_min(occupiedMinDistInt, d);
                        }
                        Ax += 10;
                    }
                    // Go to (xx1,yy++)
                    mapPtr += incrAfterRow;
                    Ay += 10;
                }

                occupiedMinDist = occupiedMinDistInt * constDist2DiscrUnits_INV ;
            }

            thisLik = zRandomTerm  + zHit * exp( Q * occupiedMinDist );
            precomputedLikelihood[ cx+cy*size_x ] = thisLik;
        }
    }

	MRPT_END
}

/*---------------------------------------------------------------
					computeLikelihoodField_II
 ---------------------------------------------------------------*/
double	 COccupancyGridMap2D::computeLikelihoodField_II( const CPointsMap	*pm, const CPose2D *relativePose )
{
	MRPT_START

	double		ret;
	size_t		N = pm->size();

	if (!N) return 1e-100; // No way to estimate this likelihood!!

	// Compute the likelihoods for each point:
	ret = 0;
//	if (likelihoodOptions.LF_alternateAverageMethod)
//			ret = 0;
//	else	ret = 1;

	TPoint2D	pointLocal,pointGlobal;

	float		zRandomTerm = 1.0f / likelihoodOptions.LF_maxRange;
	float		Q = -0.5f / square( likelihoodOptions.LF_stdHit );

	// Aux. cell indixes variables:
	int			cx,cy;
	size_t		j;
	int			cx0,cy0;
	int			cx_min, cx_max;
	int			cy_min, cy_max;
	int			maxRangeInCells = (int)ceil(likelihoodOptions.LF_maxCorrsDistance / resolution);
	int			nCells = 0;

	// -----------------------------------------------------
	// Compute around a window of neigbors around each point
	// -----------------------------------------------------
	for (j=0;j<N;j+= likelihoodOptions.LF_decimation)
	{
		// Get the point and pass it to global coordinates:
		// ---------------------------------------------
		if (relativePose)
		{
			pm->getPoint(j,pointLocal);
			pointGlobal = *relativePose + pointLocal;
		}
		else
		{
			pm->getPoint(j,pointGlobal);
		}

		// Point to cell indixes:
		// ---------------------------------------------
		cx0 = x2idx( pointGlobal.x );
		cy0 = y2idx( pointGlobal.y );

		// Compute the range of cells to compute:
		// ---------------------------------------------
		cx_min = max( cx0-maxRangeInCells,0);
		cx_max = min( cx0+maxRangeInCells,static_cast<int>(size_x));
		cy_min = max( cy0-maxRangeInCells,0);
		cy_max = min( cy0+maxRangeInCells,static_cast<int>(size_y));

//		debugImg.rectangle(cx_min,cy_min,cx_max,cy_max,0xFF0000 );

		// Compute over the window of cells:
		// ---------------------------------------------
		double  lik = 0;
		for (cx=cx_min;cx<=cx_max;cx++)
		{
			for (cy=cy_min;cy<=cy_max;cy++)
			{
				float	P_free = getCell(cx,cy);
				float	termDist = exp(Q*(square(idx2x(cx)-pointGlobal.x)+square(idx2y(cy)-pointGlobal.y) ));

				lik += P_free	  * zRandomTerm +
					   (1-P_free) * termDist;
			} // end for cy
		} // end for cx

		// Update the likelihood:
		if (likelihoodOptions.LF_alternateAverageMethod)
				ret += lik;
		else	ret += log(lik/((cy_max-cy_min+1)*(cx_max-cx_min+1)));
		nCells++;

	} // end of for each point in the scan

	if (likelihoodOptions.LF_alternateAverageMethod && nCells>0)
			ret = log(ret/nCells);
	else	ret = ret/nCells;

	return ret;

	MRPT_END
}



/*---------------------------------------------------------------
	Initilization of values, don't needed to be called directly.
  ---------------------------------------------------------------*/
COccupancyGridMap2D::TLikelihoodOptions::TLikelihoodOptions() :
	likelihoodMethod				( lmLikelihoodField_Thrun),

	LF_stdHit						( 0.35f ),
	LF_zHit							( 0.95f ),
	LF_zRandom						( 0.05f ),
	LF_maxRange						( 81.0f ),
	LF_decimation					( 5 ),
	LF_maxCorrsDistance				( 0.3f ),
	LF_alternateAverageMethod		( false ),

	MI_exponent						( 2.5f ),
	MI_skip_rays					( 10 ),
	MI_ratio_max_distance			( 1.5f ),

	rayTracing_useDistanceFilter	( true ),
	rayTracing_decimation			( 10 ),
	rayTracing_stdHit				( 1.0f ),

	consensus_takeEachRange			( 1 ),
	consensus_pow					( 5 ),
	OWA_weights						(100,1/100.0f),

	enableLikelihoodCache           ( true )
{
}

/*---------------------------------------------------------------
					loadFromConfigFile
  ---------------------------------------------------------------*/
void  COccupancyGridMap2D::TLikelihoodOptions::loadFromConfigFile(
	const mrpt::utils::CConfigFileBase  &iniFile,
	const std::string &section)
{
	MRPT_LOAD_CONFIG_VAR_CAST(likelihoodMethod, int, TLikelihoodMethod, iniFile, section);

    enableLikelihoodCache               = iniFile.read_bool(section,"enableLikelihoodCache",enableLikelihoodCache);

	LF_stdHit							= iniFile.read_float(section,"LF_stdHit",LF_stdHit);
	LF_zHit								= iniFile.read_float(section,"LF_zHit",LF_zHit);
	LF_zRandom							= iniFile.read_float(section,"LF_zRandom",LF_zRandom);
	LF_maxRange							= iniFile.read_float(section,"LF_maxRange",LF_maxRange);
	LF_decimation						= iniFile.read_int(section,"LF_decimation",LF_decimation);
	LF_maxCorrsDistance					= iniFile.read_float(section,"LF_maxCorrsDistance",LF_maxCorrsDistance);
	LF_alternateAverageMethod			= iniFile.read_bool(section,"LF_alternateAverageMethod",LF_alternateAverageMethod);

	MI_exponent							= iniFile.read_float(section,"MI_exponent",MI_exponent);
	MI_skip_rays						= iniFile.read_int(section,"MI_skip_rays",MI_skip_rays);
	MI_ratio_max_distance				= iniFile.read_float(section,"MI_ratio_max_distance",MI_ratio_max_distance);

	rayTracing_useDistanceFilter		= iniFile.read_bool(section,"rayTracing_useDistanceFilter",rayTracing_useDistanceFilter);
	rayTracing_stdHit					= iniFile.read_float(section,"rayTracing_stdHit",rayTracing_stdHit);

	consensus_takeEachRange				= iniFile.read_int(section,"consensus_takeEachRange",consensus_takeEachRange);
	consensus_pow						= iniFile.read_float(section,"consensus_pow",consensus_pow);

	iniFile.read_vector(section,"OWA_weights",OWA_weights,OWA_weights);
}

/*---------------------------------------------------------------
					dumpToTextStream
  ---------------------------------------------------------------*/
void  COccupancyGridMap2D::TLikelihoodOptions::dumpToTextStream(CStream	&out) const
{
	out.printf("\n----------- [COccupancyGridMap2D::TLikelihoodOptions] ------------ \n\n");

	out.printf("likelihoodMethod                        = ");
	switch (likelihoodMethod)
	{
	case lmMeanInformation: out.printf("lmMeanInformation"); break;
	case lmRayTracing: out.printf("lmRayTracing"); break;
	case lmConsensus: out.printf("lmConsensus"); break;
	case lmCellsDifference: out.printf("lmCellsDifference"); break;
	case lmLikelihoodField_Thrun: out.printf("lmLikelihoodField_Thrun"); break;
	case lmLikelihoodField_II: out.printf("lmLikelihoodField_II"); break;
	case lmConsensusOWA: out.printf("lmConsensusOWA"); break;
	default:
		out.printf("UNKNOWN!!!"); break;
	}
	out.printf("\n");

	out.printf("enableLikelihoodCache                   = %c\n",	enableLikelihoodCache ? 'Y':'N');

	out.printf("LF_stdHit                               = %f\n",	LF_stdHit );
	out.printf("LF_zHit                                 = %f\n",	LF_zHit );
	out.printf("LF_zRandom                              = %f\n",	LF_zRandom );
	out.printf("LF_maxRange                             = %f\n",	LF_maxRange );
	out.printf("LF_decimation                           = %u\n",	LF_decimation );
	out.printf("LF_maxCorrsDistance                     = %f\n",	LF_maxCorrsDistance );
	out.printf("LF_alternateAverageMethod               = %c\n",	LF_alternateAverageMethod ? 'Y':'N');
	out.printf("MI_exponent                             = %f\n",	MI_exponent );
	out.printf("MI_skip_rays                            = %u\n",	MI_skip_rays );
	out.printf("MI_ratio_max_distance                   = %f\n",	MI_ratio_max_distance );
	out.printf("rayTracing_useDistanceFilter            = %c\n",	rayTracing_useDistanceFilter ? 'Y':'N');
	out.printf("rayTracing_decimation                   = %u\n",	rayTracing_decimation );
	out.printf("rayTracing_stdHit                       = %f\n",	rayTracing_stdHit );
	out.printf("consensus_takeEachRange                 = %u\n",	consensus_takeEachRange );
	out.printf("consensus_pow                           = %.02f\n", consensus_pow);
	out.printf("OWA_weights   = [");
	for (size_t i=0;i<OWA_weights.size();i++)
	{
		if (i<3 || i>(OWA_weights.size()-3))
			out.printf("%.03f ",OWA_weights[i]);
		else if (i==3 && OWA_weights.size()>6)
			out.printf(" ... ");
	}
	out.printf("] (size=%u)\n",(unsigned)OWA_weights.size());
	out.printf("\n");
}

/** Returns true if this map is able to compute a sensible likelihood function for this observation (i.e. an occupancy grid map cannot with an image).
 * \param obs The observation.
 * \sa computeObservationLikelihood
 */
bool COccupancyGridMap2D::canComputeObservationLikelihood( const CObservation *obs )
{
	// Ignore laser scans if they are not planar or they are not
	//  at the altitude of this grid map:
	if ( obs->GetRuntimeClass() == CLASS_ID(CObservation2DRangeScan) )
	{
		const CObservation2DRangeScan		*scan = static_cast<const CObservation2DRangeScan*>( obs );

		if (!scan->isPlanarScan(insertionOptions.horizontalTolerance))
			return false;
		if (insertionOptions.useMapAltitude &&
			fabs(insertionOptions.mapAltitude - scan->sensorPose.z() ) > 0.01 )
			return false;

		// OK, go on...
		return true;
	}
	else // Is not a laser scanner...
	{
		return false;
	}
}

///sara add
#define LF_maxCorrsDistance_coarse  0.20f //0.20f
#define LF_stdHit_coarse            0.15f //0.15f
#define COARSE_SEARCH_SIZE          4 //4

#define LF_maxCorrsDistance_fine    0.1f //0.1f
#define LF_stdHit_fine              0.05f //0.05f
#define FINE_SEARCH_SIZE            2 //2

#define cell_value                  0.35f
#define BLACK_VALUE                 0.3f

void	 COccupancyGridMap2D::precomputedLikelihood_coarse(const int            &x_min,
                                                            const int            &y_min,
                                                            const int            &x_max,
                                                            const int            &y_max,
                                                            std::vector<float>   &likelihood)
{
	MRPT_START

    int	 K = 1;

	unsigned int	size_x_1 = size_x-1;
	unsigned int	size_y_1 = size_y-1;

	double	thisLik;
	bool black_flag = false;
	double min_value;

    for(int cx = x_min; cx < x_max; cx++)
    {
        for(int cy = y_min; cy < y_max; cy++)
        {
            min_value = 1.0;
            thisLik = 0.5;
            black_flag = false;

            int xx1 = max(0,cx-K);
            int xx2 = min(size_x_1,(unsigned)(cx+K));
            int yy1 = max(0,cy-K);
            int yy2 = min(size_y_1,(unsigned)(cy+K));

            for (int yy=yy1;yy<=yy2;yy++)
            {
                for (int xx=xx1;xx<=xx2;xx++)
                {
                    if (map[xx+yy*size_x] < p2l(0.35))
                    {
                        black_flag = true;
                        if(l2p(map[xx+yy*size_x]) < min_value)
                        {
                            min_value = l2p(map[xx+yy*size_x]);
                        }
                    }
                }
            }

            if(black_flag)
            {
                if(map[cx+cy*size_x] < p2l(0.45))
                {
                    thisLik = 0.99;
                }
                else
                {
                    thisLik = 0.75;
                }
            }
            else
            {
//                if(map[cx+cy*size_x] > p2l(0.55))
//                {
                    thisLik = 0.4;//0.4
//                }
//                else
//                {
//                    thisLik = 0.3;//0.401
//                }
            }

            likelihood[(cx - x_min) + (cy - y_min) * size_x] = thisLik;
        }
    }

	MRPT_END
}

void	 COccupancyGridMap2D::precomputedLikelihood_fine(const int            &x_min,
                                                            const int            &y_min,
                                                            const int            &x_max,
                                                            const int            &y_max,
                                                            std::vector<float>   &likelihood)
{
	MRPT_START

    int	 K = 1;

	unsigned int	size_x_1 = size_x-1;
	unsigned int	size_y_1 = size_y-1;

	double	thisLik;
	bool black_flag = false;
	double min_value;

    for(int cx = x_min; cx < x_max; cx++)
    {
        for(int cy = y_min; cy < y_max; cy++)
        {
            min_value = 1.0;
            thisLik = 0.5;
            black_flag = false;

            int xx1 = max(0,cx-K);
            int xx2 = min(size_x_1,(unsigned)(cx+K));
            int yy1 = max(0,cy-K);
            int yy2 = min(size_y_1,(unsigned)(cy+K));

            for (int yy=yy1;yy<=yy2;yy++)
            {
                for (int xx=xx1;xx<=xx2;xx++)
                {
                    if (map[xx+yy*size_x] < p2l(0.35))
                    {
                        black_flag = true;
                        if(l2p(map[xx+yy*size_x]) < min_value)
                        {
                            min_value = l2p(map[xx+yy*size_x]);
                        }
                    }
                }
            }

            if(black_flag)
            {
                if(map[cx+cy*size_x] < p2l(0.45))
                {
                    thisLik = 0.99;
                }
                else
                {
                    thisLik = 0.75;
                }
            }
            else
            {
                if(map[cx+cy*size_x] > p2l(0.7))
                {
                    thisLik = 0.4;
                }
                else
                {
                    thisLik = 0.3;//0.3
                }
            }

            likelihood[(cx - x_min) + (cy - y_min) * size_x] = thisLik;
        }
    }

	MRPT_END
}

void COccupancyGridMap2D::precomputedLikelihood_coarse1(const int &x_min, const int &y_min, const int &x_max, const int &y_max, std::vector<float> &likelihood)
{
	MRPT_START
	precomputedLikelihood_coarse(x_min, y_min, x_max, y_max, likelihood);
	MRPT_END
}
void COccupancyGridMap2D::precomputedLikelihood_coarse2(const int &x_min, const int &y_min, const int &x_max, const int &y_max, std::vector<float> &likelihood)
{
	MRPT_START
	precomputedLikelihood_coarse(x_min, y_min, x_max, y_max, likelihood);
	MRPT_END
}
void COccupancyGridMap2D::precomputedLikelihood_coarse3(const int &x_min, const int &y_min, const int &x_max, const int &y_max, std::vector<float> &likelihood)
{
	MRPT_START
	precomputedLikelihood_coarse(x_min, y_min, x_max, y_max, likelihood);
	MRPT_END
}
void COccupancyGridMap2D::precomputedLikelihood_coarse4(const int &x_min, const int &y_min, const int &x_max, const int &y_max, std::vector<float> &likelihood)
{
	MRPT_START
	precomputedLikelihood_coarse(x_min, y_min, x_max, y_max, likelihood);
	MRPT_END
}
void COccupancyGridMap2D::precomputedLikelihood_coarse5(const int &x_min, const int &y_min, const int &x_max, const int &y_max, std::vector<float> &likelihood)
{
	MRPT_START
	precomputedLikelihood_coarse(x_min, y_min, x_max, y_max, likelihood);
	MRPT_END
}
void COccupancyGridMap2D::precomputedLikelihood_coarse6(const int &x_min, const int &y_min, const int &x_max, const int &y_max, std::vector<float> &likelihood)
{
	MRPT_START
	precomputedLikelihood_coarse(x_min, y_min, x_max, y_max, likelihood);
	MRPT_END
}
void COccupancyGridMap2D::precomputedLikelihood_coarse7(const int &x_min, const int &y_min, const int &x_max, const int &y_max, std::vector<float> &likelihood)
{
	MRPT_START
	precomputedLikelihood_coarse(x_min, y_min, x_max, y_max, likelihood);
	MRPT_END
}
void COccupancyGridMap2D::precomputedLikelihood_coarse8(const int &x_min, const int &y_min, const int &x_max, const int &y_max, std::vector<float> &likelihood)
{
	MRPT_START
	precomputedLikelihood_coarse(x_min, y_min, x_max, y_max, likelihood);
	MRPT_END
}

void COccupancyGridMap2D::precomputedLikelihood_fine1(const int &x_min, const int &y_min, const int &x_max, const int &y_max, std::vector<float> &likelihood)
{
	MRPT_START
	precomputedLikelihood_fine(x_min, y_min, x_max, y_max, likelihood);
	MRPT_END
}
void COccupancyGridMap2D::precomputedLikelihood_fine2(const int &x_min, const int &y_min, const int &x_max, const int &y_max, std::vector<float> &likelihood)
{
	MRPT_START
	precomputedLikelihood_fine(x_min, y_min, x_max, y_max, likelihood);
	MRPT_END
}
void COccupancyGridMap2D::precomputedLikelihood_fine3(const int &x_min, const int &y_min, const int &x_max, const int &y_max, std::vector<float> &likelihood)
{
	MRPT_START
	precomputedLikelihood_fine(x_min, y_min, x_max, y_max, likelihood);
	MRPT_END
}
void COccupancyGridMap2D::precomputedLikelihood_fine4(const int &x_min, const int &y_min, const int &x_max, const int &y_max, std::vector<float> &likelihood)
{
	MRPT_START
	precomputedLikelihood_fine(x_min, y_min, x_max, y_max, likelihood);
	MRPT_END
}
void COccupancyGridMap2D::precomputedLikelihood_fine5(const int &x_min, const int &y_min, const int &x_max, const int &y_max, std::vector<float> &likelihood)
{
	MRPT_START
	precomputedLikelihood_fine(x_min, y_min, x_max, y_max, likelihood);
	MRPT_END
}
void COccupancyGridMap2D::precomputedLikelihood_fine6(const int &x_min, const int &y_min, const int &x_max, const int &y_max, std::vector<float> &likelihood)
{
	MRPT_START
	precomputedLikelihood_fine(x_min, y_min, x_max, y_max, likelihood);
	MRPT_END
}
void COccupancyGridMap2D::precomputedLikelihood_fine7(const int &x_min, const int &y_min, const int &x_max, const int &y_max, std::vector<float> &likelihood)
{
	MRPT_START
	precomputedLikelihood_fine(x_min, y_min, x_max, y_max, likelihood);
	MRPT_END
}
void COccupancyGridMap2D::precomputedLikelihood_fine8(const int &x_min, const int &y_min, const int &x_max, const int &y_max, std::vector<float> &likelihood)
{
	MRPT_START
	precomputedLikelihood_fine(x_min, y_min, x_max, y_max, likelihood);
	MRPT_END
}

/*void	 COccupancyGridMap2D::precomputedLikelihood_coarse1(const int            &x_min,
                                                            const int            &y_min,
                                                            const int            &x_max,
                                                            const int            &y_max,
                                                            std::vector<float>   &likelihood)
{
	MRPT_START
//	int		    K = (int)ceil(LF_maxCorrsDistance_coarse / resolution);	// The size of the checking area for matchings:
	int		    K = COARSE_SEARCH_SIZE;	// The size of the checking area for matchings:
	bool		Product_T_OrSum_F = !likelihoodOptions.LF_alternateAverageMethod;

	float		stdHit	= LF_stdHit_coarse;
	float		zHit	= likelihoodOptions.LF_zHit;
	float		zRandom	= likelihoodOptions.LF_zRandom;
	float		zRandomMaxRange	= likelihoodOptions.LF_maxRange;
	float		zRandomTerm = zRandom / zRandomMaxRange;
	float		Q = -0.5f / square(stdHit);

	unsigned int	size_x_1 = size_x-1;
	unsigned int	size_y_1 = size_y-1;

	// Aux. variables for the "for j" loop:
	double		thisLik;
	double		maxCorrDist_sq = square(LF_maxCorrsDistance_coarse);
	float		occupiedMinDist;

	cellType	thresholdCellValue = p2l(cell_value);

	const double _resolution = this->resolution;
	const double constDist2DiscrUnits = 100 / (_resolution * _resolution);
	const double constDist2DiscrUnits_INV = 1.0 / constDist2DiscrUnits;

	int cx;
	int cy;
	bool flag = true;

    for(cx = x_min; cx < x_max; cx++)
    {
        for(cy = y_min; cy < y_max; cy++)
        {
            flag = true;
            occupiedMinDist = maxCorrDist_sq; // The max.distance

            // Compute now:
            // -------------
            // Find the closest occupied cell in a certain range, given by K:
            int xx1 = max(0,cx-K);
            int xx2 = min(size_x_1,(unsigned)(cx+K));
            int yy1 = max(0,cy-K);
            int yy2 = min(size_y_1,(unsigned)(cy+K));

            // Optimized code: this part will be invoked a *lot* of times:
            {
                cellType  *mapPtr  = &map[xx1+yy1*size_x]; // Initial pointer position
                unsigned   incrAfterRow = size_x - ((xx2-xx1)+1);

                signed int Ax0 = 10*(xx1-cx);
                signed int Ay  = 10*(yy1-cy);

                unsigned int occupiedMinDistInt = mrpt::utils::round( maxCorrDist_sq * constDist2DiscrUnits );

                for (int yy=yy1;yy<=yy2;yy++)
                {
                    unsigned int Ay2 = square((unsigned int)(Ay)); // Square is faster with unsigned.
                    signed short Ax=Ax0;
                    cellType  cell;

                    for (int xx=xx1;xx<=xx2;xx++)
                    {
                        if ( (cell =*mapPtr++) < thresholdCellValue )
                        {
                            unsigned int d = square((unsigned int)(Ax)) + Ay2;
                            keep_min(occupiedMinDistInt, d);
                            flag = false;
                        }
                        Ax += 10;
                    }
                    // Go to (xx1,yy++)
                    mapPtr += incrAfterRow;
                    Ay += 10;
                }

                occupiedMinDist = occupiedMinDistInt * constDist2DiscrUnits_INV ;
            }

            thisLik = zRandomTerm  + zHit * exp( Q * occupiedMinDist );
            if(map[cx+cy*size_x] > p2l(0.7) && flag)
            {
                thisLik = BLACK_VALUE;
            }
            likelihood[(cx - x_min) + (cy - y_min) * size_x] = thisLik;
        }
    }

	MRPT_END
}

void	 COccupancyGridMap2D::precomputedLikelihood_coarse2(const int            &x_min,
                                                            const int            &y_min,
                                                            const int            &x_max,
                                                            const int            &y_max,
                                                            std::vector<float>   &likelihood)
{
	MRPT_START
    int		    K = COARSE_SEARCH_SIZE;
	bool		Product_T_OrSum_F = !likelihoodOptions.LF_alternateAverageMethod;

	float		stdHit	= LF_stdHit_coarse;
	float		zHit	= likelihoodOptions.LF_zHit;
	float		zRandom	= likelihoodOptions.LF_zRandom;
	float		zRandomMaxRange	= likelihoodOptions.LF_maxRange;
	float		zRandomTerm = zRandom / zRandomMaxRange;
	float		Q = -0.5f / square(stdHit);

	unsigned int	size_x_1 = size_x-1;
	unsigned int	size_y_1 = size_y-1;

	// Aux. variables for the "for j" loop:
	double		thisLik;
	double		maxCorrDist_sq = square(LF_maxCorrsDistance_coarse);
	float		occupiedMinDist;

	cellType	thresholdCellValue = p2l(cell_value);

	const double _resolution = this->resolution;
	const double constDist2DiscrUnits = 100 / (_resolution * _resolution);
	const double constDist2DiscrUnits_INV = 1.0 / constDist2DiscrUnits;

	int cx;
	int cy;
	bool flag = true;

    for(cx = x_min; cx < x_max; cx++)
    {
        for(cy = y_min; cy < y_max; cy++)
        {
            flag = true;
            occupiedMinDist = maxCorrDist_sq; // The max.distance

            // Compute now:
            // -------------
            // Find the closest occupied cell in a certain range, given by K:
            int xx1 = max(0,cx-K);
            int xx2 = min(size_x_1,(unsigned)(cx+K));
            int yy1 = max(0,cy-K);
            int yy2 = min(size_y_1,(unsigned)(cy+K));

            // Optimized code: this part will be invoked a *lot* of times:
            {
                cellType  *mapPtr  = &map[xx1+yy1*size_x]; // Initial pointer position
                unsigned   incrAfterRow = size_x - ((xx2-xx1)+1);

                signed int Ax0 = 10*(xx1-cx);
                signed int Ay  = 10*(yy1-cy);

                unsigned int occupiedMinDistInt = mrpt::utils::round( maxCorrDist_sq * constDist2DiscrUnits );

                for (int yy=yy1;yy<=yy2;yy++)
                {
                    unsigned int Ay2 = square((unsigned int)(Ay)); // Square is faster with unsigned.
                    signed short Ax=Ax0;
                    cellType  cell;

                    for (int xx=xx1;xx<=xx2;xx++)
                    {
                        if ( (cell =*mapPtr++) < thresholdCellValue )
                        {
                            unsigned int d = square((unsigned int)(Ax)) + Ay2;
                            keep_min(occupiedMinDistInt, d);
                            flag = false;
                        }
                        Ax += 10;
                    }
                    // Go to (xx1,yy++)
                    mapPtr += incrAfterRow;
                    Ay += 10;
                }

                occupiedMinDist = occupiedMinDistInt * constDist2DiscrUnits_INV ;
            }

            thisLik = zRandomTerm  + zHit * exp( Q * occupiedMinDist );
            if(map[cx+cy*size_x] > p2l(0.7) && flag)
            {
                thisLik = BLACK_VALUE;
            }
            likelihood[(cx - x_min) + (cy - y_min) * size_x] = thisLik;
        }
    }

	MRPT_END
}

void	 COccupancyGridMap2D::precomputedLikelihood_coarse3(const int            &x_min,
                                                            const int            &y_min,
                                                            const int            &x_max,
                                                            const int            &y_max,
                                                            std::vector<float>   &likelihood)
{
	MRPT_START
    int		    K = COARSE_SEARCH_SIZE;
	bool		Product_T_OrSum_F = !likelihoodOptions.LF_alternateAverageMethod;

	float		stdHit	= LF_stdHit_coarse;
	float		zHit	= likelihoodOptions.LF_zHit;
	float		zRandom	= likelihoodOptions.LF_zRandom;
	float		zRandomMaxRange	= likelihoodOptions.LF_maxRange;
	float		zRandomTerm = zRandom / zRandomMaxRange;
	float		Q = -0.5f / square(stdHit);

	unsigned int	size_x_1 = size_x-1;
	unsigned int	size_y_1 = size_y-1;

	// Aux. variables for the "for j" loop:
	double		thisLik;
	double		maxCorrDist_sq = square(LF_maxCorrsDistance_coarse);
	float		occupiedMinDist;

	cellType	thresholdCellValue = p2l(cell_value);

	const double _resolution = this->resolution;
	const double constDist2DiscrUnits = 100 / (_resolution * _resolution);
	const double constDist2DiscrUnits_INV = 1.0 / constDist2DiscrUnits;

	int cx;
	int cy;
	bool flag = true;

    for(cx = x_min; cx < x_max; cx++)
    {
        for(cy = y_min; cy < y_max; cy++)
        {
            flag = true;
            occupiedMinDist = maxCorrDist_sq; // The max.distance

            // Compute now:
            // -------------
            // Find the closest occupied cell in a certain range, given by K:
            int xx1 = max(0,cx-K);
            int xx2 = min(size_x_1,(unsigned)(cx+K));
            int yy1 = max(0,cy-K);
            int yy2 = min(size_y_1,(unsigned)(cy+K));

            // Optimized code: this part will be invoked a *lot* of times:
            {
                cellType  *mapPtr  = &map[xx1+yy1*size_x]; // Initial pointer position
                unsigned   incrAfterRow = size_x - ((xx2-xx1)+1);

                signed int Ax0 = 10*(xx1-cx);
                signed int Ay  = 10*(yy1-cy);

                unsigned int occupiedMinDistInt = mrpt::utils::round( maxCorrDist_sq * constDist2DiscrUnits );

                for (int yy=yy1;yy<=yy2;yy++)
                {
                    unsigned int Ay2 = square((unsigned int)(Ay)); // Square is faster with unsigned.
                    signed short Ax=Ax0;
                    cellType  cell;

                    for (int xx=xx1;xx<=xx2;xx++)
                    {
                        if ( (cell =*mapPtr++) < thresholdCellValue )
                        {
                            unsigned int d = square((unsigned int)(Ax)) + Ay2;
                            keep_min(occupiedMinDistInt, d);
                            flag = false;
                        }
                        Ax += 10;
                    }
                    // Go to (xx1,yy++)
                    mapPtr += incrAfterRow;
                    Ay += 10;
                }

                occupiedMinDist = occupiedMinDistInt * constDist2DiscrUnits_INV ;
            }

            thisLik = zRandomTerm  + zHit * exp( Q * occupiedMinDist );
            if(map[cx+cy*size_x] > p2l(0.7) && flag)
            {
                thisLik = BLACK_VALUE;
            }
            likelihood[(cx - x_min) + (cy - y_min) * size_x] = thisLik;
        }
    }

	MRPT_END
}

void	 COccupancyGridMap2D::precomputedLikelihood_coarse4(const int            &x_min,
                                                            const int            &y_min,
                                                            const int            &x_max,
                                                            const int            &y_max,
                                                            std::vector<float>   &likelihood)
{
	MRPT_START
	int		    K = COARSE_SEARCH_SIZE;
	bool		Product_T_OrSum_F = !likelihoodOptions.LF_alternateAverageMethod;

	float		stdHit	= LF_stdHit_coarse;
	float		zHit	= likelihoodOptions.LF_zHit;
	float		zRandom	= likelihoodOptions.LF_zRandom;
	float		zRandomMaxRange	= likelihoodOptions.LF_maxRange;
	float		zRandomTerm = zRandom / zRandomMaxRange;
	float		Q = -0.5f / square(stdHit);

	unsigned int	size_x_1 = size_x-1;
	unsigned int	size_y_1 = size_y-1;

	// Aux. variables for the "for j" loop:
	double		thisLik;
	double		maxCorrDist_sq = square(LF_maxCorrsDistance_coarse);
	float		occupiedMinDist;

	cellType	thresholdCellValue = p2l(cell_value);

	const double _resolution = this->resolution;
	const double constDist2DiscrUnits = 100 / (_resolution * _resolution);
	const double constDist2DiscrUnits_INV = 1.0 / constDist2DiscrUnits;

	int cx;
	int cy;
	bool flag = true;

    for(cx = x_min; cx < x_max; cx++)
    {
        for(cy = y_min; cy < y_max; cy++)
        {
            flag = true;

            occupiedMinDist = maxCorrDist_sq; // The max.distance

            // Compute now:
            // -------------
            // Find the closest occupied cell in a certain range, given by K:
            int xx1 = max(0,cx-K);
            int xx2 = min(size_x_1,(unsigned)(cx+K));
            int yy1 = max(0,cy-K);
            int yy2 = min(size_y_1,(unsigned)(cy+K));

            // Optimized code: this part will be invoked a *lot* of times:
            {
                cellType  *mapPtr  = &map[xx1+yy1*size_x]; // Initial pointer position
                unsigned   incrAfterRow = size_x - ((xx2-xx1)+1);

                signed int Ax0 = 10*(xx1-cx);
                signed int Ay  = 10*(yy1-cy);

                unsigned int occupiedMinDistInt = mrpt::utils::round( maxCorrDist_sq * constDist2DiscrUnits );

                for (int yy=yy1;yy<=yy2;yy++)
                {
                    unsigned int Ay2 = square((unsigned int)(Ay)); // Square is faster with unsigned.
                    signed short Ax=Ax0;
                    cellType  cell;

                    for (int xx=xx1;xx<=xx2;xx++)
                    {
                        if ( (cell =*mapPtr++) < thresholdCellValue )
                        {
                            unsigned int d = square((unsigned int)(Ax)) + Ay2;
                            keep_min(occupiedMinDistInt, d);
                            flag = false;
                        }
                        Ax += 10;
                    }
                    // Go to (xx1,yy++)
                    mapPtr += incrAfterRow;
                    Ay += 10;
                }

                occupiedMinDist = occupiedMinDistInt * constDist2DiscrUnits_INV ;
            }

            thisLik = zRandomTerm  + zHit * exp( Q * occupiedMinDist );
            if(l2p(map[cx+cy*size_x]) > 0.7 && flag)
            {
                thisLik = BLACK_VALUE;
            }
            likelihood[(cx - x_min) + (cy - y_min) * size_x] = thisLik;
        }
    }

	MRPT_END
}

void	 COccupancyGridMap2D::precomputedLikelihood_coarse5(const int            &x_min,
                                                            const int            &y_min,
                                                            const int            &x_max,
                                                            const int            &y_max,
                                                            std::vector<float>   &likelihood)
{
	MRPT_START
	int		    K = COARSE_SEARCH_SIZE;
	bool		Product_T_OrSum_F = !likelihoodOptions.LF_alternateAverageMethod;

	float		stdHit	= LF_stdHit_coarse;
	float		zHit	= likelihoodOptions.LF_zHit;
	float		zRandom	= likelihoodOptions.LF_zRandom;
	float		zRandomMaxRange	= likelihoodOptions.LF_maxRange;
	float		zRandomTerm = zRandom / zRandomMaxRange;
	float		Q = -0.5f / square(stdHit);

	unsigned int	size_x_1 = size_x-1;
	unsigned int	size_y_1 = size_y-1;

	// Aux. variables for the "for j" loop:
	double		thisLik;
	double		maxCorrDist_sq = square(LF_maxCorrsDistance_coarse);
	float		occupiedMinDist;

	cellType	thresholdCellValue = p2l(cell_value);

	const double _resolution = this->resolution;
	const double constDist2DiscrUnits = 100 / (_resolution * _resolution);
	const double constDist2DiscrUnits_INV = 1.0 / constDist2DiscrUnits;

	int cx;
	int cy;
	bool flag = true;

    for(cx = x_min; cx < x_max; cx++)
    {
        for(cy = y_min; cy < y_max; cy++)
        {
            flag = true;
            occupiedMinDist = maxCorrDist_sq; // The max.distance

            // Compute now:
            // -------------
            // Find the closest occupied cell in a certain range, given by K:
            int xx1 = max(0,cx-K);
            int xx2 = min(size_x_1,(unsigned)(cx+K));
            int yy1 = max(0,cy-K);
            int yy2 = min(size_y_1,(unsigned)(cy+K));

            // Optimized code: this part will be invoked a *lot* of times:
            {
                cellType  *mapPtr  = &map[xx1+yy1*size_x]; // Initial pointer position
                unsigned   incrAfterRow = size_x - ((xx2-xx1)+1);

                signed int Ax0 = 10*(xx1-cx);
                signed int Ay  = 10*(yy1-cy);

                unsigned int occupiedMinDistInt = mrpt::utils::round( maxCorrDist_sq * constDist2DiscrUnits );

                for (int yy=yy1;yy<=yy2;yy++)
                {
                    unsigned int Ay2 = square((unsigned int)(Ay)); // Square is faster with unsigned.
                    signed short Ax=Ax0;
                    cellType  cell;

                    for (int xx=xx1;xx<=xx2;xx++)
                    {
                        if ( (cell =*mapPtr++) < thresholdCellValue )
                        {
                            unsigned int d = square((unsigned int)(Ax)) + Ay2;
                            keep_min(occupiedMinDistInt, d);
                            flag = false;
                        }
                        Ax += 10;
                    }
                    // Go to (xx1,yy++)
                    mapPtr += incrAfterRow;
                    Ay += 10;
                }

                occupiedMinDist = occupiedMinDistInt * constDist2DiscrUnits_INV ;
            }

            thisLik = zRandomTerm  + zHit * exp( Q * occupiedMinDist );
            if(map[cx+cy*size_x] > p2l(0.7) && flag)
            {
                thisLik = BLACK_VALUE;
            }
            likelihood[(cx - x_min) + (cy - y_min) * size_x] = thisLik;
        }
    }

	MRPT_END
}

void	 COccupancyGridMap2D::precomputedLikelihood_coarse6(const int            &x_min,
                                                            const int            &y_min,
                                                            const int            &x_max,
                                                            const int            &y_max,
                                                            std::vector<float>   &likelihood)
{
	MRPT_START
	int		    K = COARSE_SEARCH_SIZE;
	bool		Product_T_OrSum_F = !likelihoodOptions.LF_alternateAverageMethod;

	float		stdHit	= LF_stdHit_coarse;
	float		zHit	= likelihoodOptions.LF_zHit;
	float		zRandom	= likelihoodOptions.LF_zRandom;
	float		zRandomMaxRange	= likelihoodOptions.LF_maxRange;
	float		zRandomTerm = zRandom / zRandomMaxRange;
	float		Q = -0.5f / square(stdHit);

	unsigned int	size_x_1 = size_x-1;
	unsigned int	size_y_1 = size_y-1;

	// Aux. variables for the "for j" loop:
	double		thisLik;
	double		maxCorrDist_sq = square(LF_maxCorrsDistance_coarse);
	float		occupiedMinDist;

	cellType	thresholdCellValue = p2l(cell_value);

	const double _resolution = this->resolution;
	const double constDist2DiscrUnits = 100 / (_resolution * _resolution);
	const double constDist2DiscrUnits_INV = 1.0 / constDist2DiscrUnits;

	int cx;
	int cy;
	bool flag = true;

    for(cx = x_min; cx < x_max; cx++)
    {
        for(cy = y_min; cy < y_max; cy++)
        {
            flag = true;
            occupiedMinDist = maxCorrDist_sq; // The max.distance

            // Compute now:
            // -------------
            // Find the closest occupied cell in a certain range, given by K:
            int xx1 = max(0,cx-K);
            int xx2 = min(size_x_1,(unsigned)(cx+K));
            int yy1 = max(0,cy-K);
            int yy2 = min(size_y_1,(unsigned)(cy+K));

            // Optimized code: this part will be invoked a *lot* of times:
            {
                cellType  *mapPtr  = &map[xx1+yy1*size_x]; // Initial pointer position
                unsigned   incrAfterRow = size_x - ((xx2-xx1)+1);

                signed int Ax0 = 10*(xx1-cx);
                signed int Ay  = 10*(yy1-cy);

                unsigned int occupiedMinDistInt = mrpt::utils::round( maxCorrDist_sq * constDist2DiscrUnits );

                for (int yy=yy1;yy<=yy2;yy++)
                {
                    unsigned int Ay2 = square((unsigned int)(Ay)); // Square is faster with unsigned.
                    signed short Ax=Ax0;
                    cellType  cell;

                    for (int xx=xx1;xx<=xx2;xx++)
                    {
                        if ( (cell =*mapPtr++) < thresholdCellValue )
                        {
                            unsigned int d = square((unsigned int)(Ax)) + Ay2;
                            keep_min(occupiedMinDistInt, d);
                            flag = false;
                        }
                        Ax += 10;
                    }
                    // Go to (xx1,yy++)
                    mapPtr += incrAfterRow;
                    Ay += 10;
                }

                occupiedMinDist = occupiedMinDistInt * constDist2DiscrUnits_INV ;
            }

            thisLik = zRandomTerm  + zHit * exp( Q * occupiedMinDist );
            if(map[cx+cy*size_x] > p2l(0.7) && flag)
            {
                thisLik = BLACK_VALUE;
            }
            likelihood[(cx - x_min) + (cy - y_min) * size_x] = thisLik;
        }
    }

	MRPT_END
}

void	 COccupancyGridMap2D::precomputedLikelihood_coarse7(const int            &x_min,
                                                            const int            &y_min,
                                                            const int            &x_max,
                                                            const int            &y_max,
                                                            std::vector<float>   &likelihood)
{
	MRPT_START
	int		    K = COARSE_SEARCH_SIZE;
	bool		Product_T_OrSum_F = !likelihoodOptions.LF_alternateAverageMethod;

	float		stdHit	= LF_stdHit_coarse;
	float		zHit	= likelihoodOptions.LF_zHit;
	float		zRandom	= likelihoodOptions.LF_zRandom;
	float		zRandomMaxRange	= likelihoodOptions.LF_maxRange;
	float		zRandomTerm = zRandom / zRandomMaxRange;
	float		Q = -0.5f / square(stdHit);

	unsigned int	size_x_1 = size_x-1;
	unsigned int	size_y_1 = size_y-1;

	// Aux. variables for the "for j" loop:
	double		thisLik;
	double		maxCorrDist_sq = square(LF_maxCorrsDistance_coarse);
	float		occupiedMinDist;

	cellType	thresholdCellValue = p2l(cell_value);

	const double _resolution = this->resolution;
	const double constDist2DiscrUnits = 100 / (_resolution * _resolution);
	const double constDist2DiscrUnits_INV = 1.0 / constDist2DiscrUnits;

	int cx;
	int cy;
	bool flag = true;

    for(cx = x_min; cx < x_max; cx++)
    {
        for(cy = y_min; cy < y_max; cy++)
        {
            flag = true;
            occupiedMinDist = maxCorrDist_sq; // The max.distance

            // Compute now:
            // -------------
            // Find the closest occupied cell in a certain range, given by K:
            int xx1 = max(0,cx-K);
            int xx2 = min(size_x_1,(unsigned)(cx+K));
            int yy1 = max(0,cy-K);
            int yy2 = min(size_y_1,(unsigned)(cy+K));

            // Optimized code: this part will be invoked a *lot* of times:
            {
                cellType  *mapPtr  = &map[xx1+yy1*size_x]; // Initial pointer position
                unsigned   incrAfterRow = size_x - ((xx2-xx1)+1);

                signed int Ax0 = 10*(xx1-cx);
                signed int Ay  = 10*(yy1-cy);

                unsigned int occupiedMinDistInt = mrpt::utils::round( maxCorrDist_sq * constDist2DiscrUnits );

                for (int yy=yy1;yy<=yy2;yy++)
                {
                    unsigned int Ay2 = square((unsigned int)(Ay)); // Square is faster with unsigned.
                    signed short Ax=Ax0;
                    cellType  cell;

                    for (int xx=xx1;xx<=xx2;xx++)
                    {
                        if ( (cell =*mapPtr++) < thresholdCellValue )
                        {
                            unsigned int d = square((unsigned int)(Ax)) + Ay2;
                            keep_min(occupiedMinDistInt, d);
                            flag = false;
                        }
                        Ax += 10;
                    }
                    // Go to (xx1,yy++)
                    mapPtr += incrAfterRow;
                    Ay += 10;
                }

                occupiedMinDist = occupiedMinDistInt * constDist2DiscrUnits_INV ;
            }

            thisLik = zRandomTerm  + zHit * exp( Q * occupiedMinDist );
            if(map[cx+cy*size_x] > p2l(0.7) && flag)
            {
                thisLik = BLACK_VALUE;
            }
            likelihood[(cx - x_min) + (cy - y_min) * size_x] = thisLik;
        }
    }

	MRPT_END
}

void	 COccupancyGridMap2D::precomputedLikelihood_coarse8(const int            &x_min,
                                                            const int            &y_min,
                                                            const int            &x_max,
                                                            const int            &y_max,
                                                            std::vector<float>   &likelihood)
{
	MRPT_START
	int		    K = COARSE_SEARCH_SIZE;
	bool		Product_T_OrSum_F = !likelihoodOptions.LF_alternateAverageMethod;

	float		stdHit	= LF_stdHit_coarse;
	float		zHit	= likelihoodOptions.LF_zHit;
	float		zRandom	= likelihoodOptions.LF_zRandom;
	float		zRandomMaxRange	= likelihoodOptions.LF_maxRange;
	float		zRandomTerm = zRandom / zRandomMaxRange;
	float		Q = -0.5f / square(stdHit);

	unsigned int	size_x_1 = size_x-1;
	unsigned int	size_y_1 = size_y-1;

	// Aux. variables for the "for j" loop:
	double		thisLik;
	double		maxCorrDist_sq = square(LF_maxCorrsDistance_coarse);
	float		occupiedMinDist;

	cellType	thresholdCellValue = p2l(cell_value);

	const double _resolution = this->resolution;
	const double constDist2DiscrUnits = 100 / (_resolution * _resolution);
	const double constDist2DiscrUnits_INV = 1.0 / constDist2DiscrUnits;

	int cx;
	int cy;
	bool flag = true;

    for(cx = x_min; cx < x_max; cx++)
    {
        for(cy = y_min; cy < y_max; cy++)
        {
            flag = true;
            occupiedMinDist = maxCorrDist_sq; // The max.distance

            // Compute now:
            // -------------
            // Find the closest occupied cell in a certain range, given by K:
            int xx1 = max(0,cx-K);
            int xx2 = min(size_x_1,(unsigned)(cx+K));
            int yy1 = max(0,cy-K);
            int yy2 = min(size_y_1,(unsigned)(cy+K));

            // Optimized code: this part will be invoked a *lot* of times:
            {
                cellType  *mapPtr  = &map[xx1+yy1*size_x]; // Initial pointer position
                unsigned   incrAfterRow = size_x - ((xx2-xx1)+1);

                signed int Ax0 = 10*(xx1-cx);
                signed int Ay  = 10*(yy1-cy);

                unsigned int occupiedMinDistInt = mrpt::utils::round( maxCorrDist_sq * constDist2DiscrUnits );

                for (int yy=yy1;yy<=yy2;yy++)
                {
                    unsigned int Ay2 = square((unsigned int)(Ay)); // Square is faster with unsigned.
                    signed short Ax=Ax0;
                    cellType  cell;

                    for (int xx=xx1;xx<=xx2;xx++)
                    {
                        if ( (cell =*mapPtr++) < thresholdCellValue )
                        {
                            unsigned int d = square((unsigned int)(Ax)) + Ay2;
                            keep_min(occupiedMinDistInt, d);
                            flag = false;
                        }
                        Ax += 10;
                    }
                    // Go to (xx1,yy++)
                    mapPtr += incrAfterRow;
                    Ay += 10;
                }

                occupiedMinDist = occupiedMinDistInt * constDist2DiscrUnits_INV ;
            }

            thisLik = zRandomTerm  + zHit * exp( Q * occupiedMinDist );
            if(map[cx+cy*size_x] > p2l(0.7) && flag)
            {
                thisLik = BLACK_VALUE;
            }
            likelihood[(cx - x_min) + (cy - y_min) * size_x] = thisLik;
        }
    }

	MRPT_END
}

void COccupancyGridMap2D::precomputedLikelihood_fine1(const int            &x_min,
                                                      const int            &y_min,
                                                      const int            &x_max,
                                                      const int            &y_max,
                                                      std::vector<float>   &likelihood)
{
	MRPT_START
//	int		    K = (int)ceil(LF_maxCorrsDistance_coarse / resolution);	// The size of the checking area for matchings:
	int		    K = FINE_SEARCH_SIZE;	// The size of the checking area for matchings:
	bool		Product_T_OrSum_F = !likelihoodOptions.LF_alternateAverageMethod;

	float		stdHit	= LF_stdHit_fine;
	float		zHit	= likelihoodOptions.LF_zHit;
	float		zRandom	= likelihoodOptions.LF_zRandom;
	float		zRandomMaxRange	= likelihoodOptions.LF_maxRange;
	float		zRandomTerm = zRandom / zRandomMaxRange;
	float		Q = -0.5f / square(stdHit);

	unsigned int	size_x_1 = size_x-1;
	unsigned int	size_y_1 = size_y-1;

	// Aux. variables for the "for j" loop:
	double		thisLik;
	double		maxCorrDist_sq = square(LF_maxCorrsDistance_fine);
	float		occupiedMinDist;

	cellType	thresholdCellValue = p2l(cell_value);

	const double _resolution = this->resolution;
	const double constDist2DiscrUnits = 100 / (_resolution * _resolution);
	const double constDist2DiscrUnits_INV = 1.0 / constDist2DiscrUnits;

	int cx;
	int cy;
	bool flag = true;

    for(cx = x_min; cx < x_max; cx++)
    {
        for(cy = y_min; cy < y_max; cy++)
        {
            flag = true;
            occupiedMinDist = maxCorrDist_sq; // The max.distance

            // Compute now:
            // -------------
            // Find the closest occupied cell in a certain range, given by K:
            int xx1 = max(0,cx-K);
            int xx2 = min(size_x_1,(unsigned)(cx+K));
            int yy1 = max(0,cy-K);
            int yy2 = min(size_y_1,(unsigned)(cy+K));

            // Optimized code: this part will be invoked a *lot* of times:
            {
                cellType  *mapPtr  = &map[xx1+yy1*size_x]; // Initial pointer position
                unsigned   incrAfterRow = size_x - ((xx2-xx1)+1);

                signed int Ax0 = 10*(xx1-cx);
                signed int Ay  = 10*(yy1-cy);

                unsigned int occupiedMinDistInt = mrpt::utils::round( maxCorrDist_sq * constDist2DiscrUnits );

                for (int yy=yy1;yy<=yy2;yy++)
                {
                    unsigned int Ay2 = square((unsigned int)(Ay)); // Square is faster with unsigned.
                    signed short Ax=Ax0;
                    cellType  cell;

                    for (int xx=xx1;xx<=xx2;xx++)
                    {
                        if ( (cell =*mapPtr++) < thresholdCellValue )
                        {
                            unsigned int d = square((unsigned int)(Ax)) + Ay2;
                            keep_min(occupiedMinDistInt, d);
                            flag = false;
                        }
                        Ax += 10;
                    }
                    // Go to (xx1,yy++)
                    mapPtr += incrAfterRow;
                    Ay += 10;
                }

                occupiedMinDist = occupiedMinDistInt * constDist2DiscrUnits_INV ;
            }

            thisLik = zRandomTerm  + zHit * exp( Q * occupiedMinDist );
            if(map[cx+cy*size_x] > p2l(0.7) && flag)
            {
                thisLik = BLACK_VALUE;
            }
            likelihood[(cx - x_min) + (cy - y_min) * size_x] = thisLik;
        }
    }

	MRPT_END
}

void COccupancyGridMap2D::precomputedLikelihood_fine2(const int            &x_min,
                                                      const int            &y_min,
                                                      const int            &x_max,
                                                      const int            &y_max,
                                                      std::vector<float>   &likelihood)
{
	MRPT_START
//	int		    K = (int)ceil(LF_maxCorrsDistance_coarse / resolution);	// The size of the checking area for matchings:
	int		    K = FINE_SEARCH_SIZE;	// The size of the checking area for matchings:
	bool		Product_T_OrSum_F = !likelihoodOptions.LF_alternateAverageMethod;

	float		stdHit	= LF_stdHit_fine;
	float		zHit	= likelihoodOptions.LF_zHit;
	float		zRandom	= likelihoodOptions.LF_zRandom;
	float		zRandomMaxRange	= likelihoodOptions.LF_maxRange;
	float		zRandomTerm = zRandom / zRandomMaxRange;
	float		Q = -0.5f / square(stdHit);

	unsigned int	size_x_1 = size_x-1;
	unsigned int	size_y_1 = size_y-1;

	// Aux. variables for the "for j" loop:
	double		thisLik;
	double		maxCorrDist_sq = square(LF_maxCorrsDistance_fine);
	float		occupiedMinDist;

	cellType	thresholdCellValue = p2l(cell_value);

	const double _resolution = this->resolution;
	const double constDist2DiscrUnits = 100 / (_resolution * _resolution);
	const double constDist2DiscrUnits_INV = 1.0 / constDist2DiscrUnits;

	int cx;
	int cy;
	bool flag = true;

    for(cx = x_min; cx < x_max; cx++)
    {
        for(cy = y_min; cy < y_max; cy++)
        {
            flag = true;
            occupiedMinDist = maxCorrDist_sq; // The max.distance

            // Compute now:
            // -------------
            // Find the closest occupied cell in a certain range, given by K:
            int xx1 = max(0,cx-K);
            int xx2 = min(size_x_1,(unsigned)(cx+K));
            int yy1 = max(0,cy-K);
            int yy2 = min(size_y_1,(unsigned)(cy+K));

            // Optimized code: this part will be invoked a *lot* of times:
            {
                cellType  *mapPtr  = &map[xx1+yy1*size_x]; // Initial pointer position
                unsigned   incrAfterRow = size_x - ((xx2-xx1)+1);

                signed int Ax0 = 10*(xx1-cx);
                signed int Ay  = 10*(yy1-cy);

                unsigned int occupiedMinDistInt = mrpt::utils::round( maxCorrDist_sq * constDist2DiscrUnits );

                for (int yy=yy1;yy<=yy2;yy++)
                {
                    unsigned int Ay2 = square((unsigned int)(Ay)); // Square is faster with unsigned.
                    signed short Ax=Ax0;
                    cellType  cell;

                    for (int xx=xx1;xx<=xx2;xx++)
                    {
                        if ( (cell =*mapPtr++) < thresholdCellValue )
                        {
                            unsigned int d = square((unsigned int)(Ax)) + Ay2;
                            keep_min(occupiedMinDistInt, d);
                            flag = false;
                        }
                        Ax += 10;
                    }
                    // Go to (xx1,yy++)
                    mapPtr += incrAfterRow;
                    Ay += 10;
                }

                occupiedMinDist = occupiedMinDistInt * constDist2DiscrUnits_INV ;
            }

            thisLik = zRandomTerm  + zHit * exp( Q * occupiedMinDist );
            if(map[cx+cy*size_x] > p2l(0.7) && flag)
            {
                thisLik = BLACK_VALUE;
            }
            likelihood[(cx - x_min) + (cy - y_min) * size_x] = thisLik;
        }
    }

	MRPT_END
}

void COccupancyGridMap2D::precomputedLikelihood_fine3(const int            &x_min,
                                                      const int            &y_min,
                                                      const int            &x_max,
                                                      const int            &y_max,
                                                      std::vector<float>   &likelihood)
{
	MRPT_START
//	int		    K = (int)ceil(LF_maxCorrsDistance_coarse / resolution);	// The size of the checking area for matchings:
	int		    K = FINE_SEARCH_SIZE;	// The size of the checking area for matchings:
	bool		Product_T_OrSum_F = !likelihoodOptions.LF_alternateAverageMethod;

	float		stdHit	= LF_stdHit_fine;
	float		zHit	= likelihoodOptions.LF_zHit;
	float		zRandom	= likelihoodOptions.LF_zRandom;
	float		zRandomMaxRange	= likelihoodOptions.LF_maxRange;
	float		zRandomTerm = zRandom / zRandomMaxRange;
	float		Q = -0.5f / square(stdHit);

	unsigned int	size_x_1 = size_x-1;
	unsigned int	size_y_1 = size_y-1;

	// Aux. variables for the "for j" loop:
	double		thisLik;
	double		maxCorrDist_sq = square(LF_maxCorrsDistance_fine);
	float		occupiedMinDist;

	cellType	thresholdCellValue = p2l(cell_value);

	const double _resolution = this->resolution;
	const double constDist2DiscrUnits = 100 / (_resolution * _resolution);
	const double constDist2DiscrUnits_INV = 1.0 / constDist2DiscrUnits;

	int cx;
	int cy;
	bool flag = true;

    for(cx = x_min; cx < x_max; cx++)
    {
        for(cy = y_min; cy < y_max; cy++)
        {
            flag = true;
            occupiedMinDist = maxCorrDist_sq; // The max.distance

            // Compute now:
            // -------------
            // Find the closest occupied cell in a certain range, given by K:
            int xx1 = max(0,cx-K);
            int xx2 = min(size_x_1,(unsigned)(cx+K));
            int yy1 = max(0,cy-K);
            int yy2 = min(size_y_1,(unsigned)(cy+K));

            // Optimized code: this part will be invoked a *lot* of times:
            {
                cellType  *mapPtr  = &map[xx1+yy1*size_x]; // Initial pointer position
                unsigned   incrAfterRow = size_x - ((xx2-xx1)+1);

                signed int Ax0 = 10*(xx1-cx);
                signed int Ay  = 10*(yy1-cy);

                unsigned int occupiedMinDistInt = mrpt::utils::round( maxCorrDist_sq * constDist2DiscrUnits );

                for (int yy=yy1;yy<=yy2;yy++)
                {
                    unsigned int Ay2 = square((unsigned int)(Ay)); // Square is faster with unsigned.
                    signed short Ax=Ax0;
                    cellType  cell;

                    for (int xx=xx1;xx<=xx2;xx++)
                    {
                        if ( (cell =*mapPtr++) < thresholdCellValue )
                        {
                            unsigned int d = square((unsigned int)(Ax)) + Ay2;
                            keep_min(occupiedMinDistInt, d);
                            flag = false;
                        }
                        Ax += 10;
                    }
                    // Go to (xx1,yy++)
                    mapPtr += incrAfterRow;
                    Ay += 10;
                }

                occupiedMinDist = occupiedMinDistInt * constDist2DiscrUnits_INV ;
            }

            thisLik = zRandomTerm  + zHit * exp( Q * occupiedMinDist );
            if(map[cx+cy*size_x] > p2l(0.7) && flag)
            {
                thisLik = BLACK_VALUE;
            }
            likelihood[(cx - x_min) + (cy - y_min) * size_x] = thisLik;
        }
    }

	MRPT_END
}

void COccupancyGridMap2D::precomputedLikelihood_fine4(const int            &x_min,
                                                      const int            &y_min,
                                                      const int            &x_max,
                                                      const int            &y_max,
                                                      std::vector<float>   &likelihood)
{
	MRPT_START
//	int		    K = (int)ceil(LF_maxCorrsDistance_coarse / resolution);	// The size of the checking area for matchings:
	int		    K = FINE_SEARCH_SIZE;	// The size of the checking area for matchings:
	bool		Product_T_OrSum_F = !likelihoodOptions.LF_alternateAverageMethod;

	float		stdHit	= LF_stdHit_fine;
	float		zHit	= likelihoodOptions.LF_zHit;
	float		zRandom	= likelihoodOptions.LF_zRandom;
	float		zRandomMaxRange	= likelihoodOptions.LF_maxRange;
	float		zRandomTerm = zRandom / zRandomMaxRange;
	float		Q = -0.5f / square(stdHit);

	unsigned int	size_x_1 = size_x-1;
	unsigned int	size_y_1 = size_y-1;

	// Aux. variables for the "for j" loop:
	double		thisLik;
	double		maxCorrDist_sq = square(LF_maxCorrsDistance_fine);
	float		occupiedMinDist;

	cellType	thresholdCellValue = p2l(cell_value);

	const double _resolution = this->resolution;
	const double constDist2DiscrUnits = 100 / (_resolution * _resolution);
	const double constDist2DiscrUnits_INV = 1.0 / constDist2DiscrUnits;

	int cx;
	int cy;
	bool flag = true;

    for(cx = x_min; cx < x_max; cx++)
    {
        for(cy = y_min; cy < y_max; cy++)
        {
            flag = true;
            occupiedMinDist = maxCorrDist_sq; // The max.distance

            // Compute now:
            // -------------
            // Find the closest occupied cell in a certain range, given by K:
            int xx1 = max(0,cx-K);
            int xx2 = min(size_x_1,(unsigned)(cx+K));
            int yy1 = max(0,cy-K);
            int yy2 = min(size_y_1,(unsigned)(cy+K));

            // Optimized code: this part will be invoked a *lot* of times:
            {
                cellType  *mapPtr  = &map[xx1+yy1*size_x]; // Initial pointer position
                unsigned   incrAfterRow = size_x - ((xx2-xx1)+1);

                signed int Ax0 = 10*(xx1-cx);
                signed int Ay  = 10*(yy1-cy);

                unsigned int occupiedMinDistInt = mrpt::utils::round( maxCorrDist_sq * constDist2DiscrUnits );

                for (int yy=yy1;yy<=yy2;yy++)
                {
                    unsigned int Ay2 = square((unsigned int)(Ay)); // Square is faster with unsigned.
                    signed short Ax=Ax0;
                    cellType  cell;

                    for (int xx=xx1;xx<=xx2;xx++)
                    {
                        if ( (cell =*mapPtr++) < thresholdCellValue )
                        {
                            unsigned int d = square((unsigned int)(Ax)) + Ay2;
                            keep_min(occupiedMinDistInt, d);
                            flag = false;
                        }
                        Ax += 10;
                    }
                    // Go to (xx1,yy++)
                    mapPtr += incrAfterRow;
                    Ay += 10;
                }

                occupiedMinDist = occupiedMinDistInt * constDist2DiscrUnits_INV ;
            }

            thisLik = zRandomTerm  + zHit * exp( Q * occupiedMinDist );
            if(map[cx+cy*size_x] > p2l(0.7) && flag)
            {
                thisLik = BLACK_VALUE;
            }
            likelihood[(cx - x_min) + (cy - y_min) * size_x] = thisLik;
        }
    }

	MRPT_END
}

void COccupancyGridMap2D::precomputedLikelihood_fine5(const int            &x_min,
                                                      const int            &y_min,
                                                      const int            &x_max,
                                                      const int            &y_max,
                                                      std::vector<float>   &likelihood)
{
	MRPT_START
//	int		    K = (int)ceil(LF_maxCorrsDistance_coarse / resolution);	// The size of the checking area for matchings:
	int		    K = FINE_SEARCH_SIZE;	// The size of the checking area for matchings:
	bool		Product_T_OrSum_F = !likelihoodOptions.LF_alternateAverageMethod;

	float		stdHit	= LF_stdHit_fine;
	float		zHit	= likelihoodOptions.LF_zHit;
	float		zRandom	= likelihoodOptions.LF_zRandom;
	float		zRandomMaxRange	= likelihoodOptions.LF_maxRange;
	float		zRandomTerm = zRandom / zRandomMaxRange;
	float		Q = -0.5f / square(stdHit);

	unsigned int	size_x_1 = size_x-1;
	unsigned int	size_y_1 = size_y-1;

	// Aux. variables for the "for j" loop:
	double		thisLik;
	double		maxCorrDist_sq = square(LF_maxCorrsDistance_fine);
	float		occupiedMinDist;

	cellType	thresholdCellValue = p2l(cell_value);

	const double _resolution = this->resolution;
	const double constDist2DiscrUnits = 100 / (_resolution * _resolution);
	const double constDist2DiscrUnits_INV = 1.0 / constDist2DiscrUnits;

	int cx;
	int cy;
	bool flag = true;

    for(cx = x_min; cx < x_max; cx++)
    {
        for(cy = y_min; cy < y_max; cy++)
        {
            flag = true;
            occupiedMinDist = maxCorrDist_sq; // The max.distance

            // Compute now:
            // -------------
            // Find the closest occupied cell in a certain range, given by K:
            int xx1 = max(0,cx-K);
            int xx2 = min(size_x_1,(unsigned)(cx+K));
            int yy1 = max(0,cy-K);
            int yy2 = min(size_y_1,(unsigned)(cy+K));

            // Optimized code: this part will be invoked a *lot* of times:
            {
                cellType  *mapPtr  = &map[xx1+yy1*size_x]; // Initial pointer position
                unsigned   incrAfterRow = size_x - ((xx2-xx1)+1);

                signed int Ax0 = 10*(xx1-cx);
                signed int Ay  = 10*(yy1-cy);

                unsigned int occupiedMinDistInt = mrpt::utils::round( maxCorrDist_sq * constDist2DiscrUnits );

                for (int yy=yy1;yy<=yy2;yy++)
                {
                    unsigned int Ay2 = square((unsigned int)(Ay)); // Square is faster with unsigned.
                    signed short Ax=Ax0;
                    cellType  cell;

                    for (int xx=xx1;xx<=xx2;xx++)
                    {
                        if ( (cell =*mapPtr++) < thresholdCellValue )
                        {
                            unsigned int d = square((unsigned int)(Ax)) + Ay2;
                            keep_min(occupiedMinDistInt, d);
                            flag = false;
                        }
                        Ax += 10;
                    }
                    // Go to (xx1,yy++)
                    mapPtr += incrAfterRow;
                    Ay += 10;
                }

                occupiedMinDist = occupiedMinDistInt * constDist2DiscrUnits_INV ;
            }

            thisLik = zRandomTerm  + zHit * exp( Q * occupiedMinDist );
            if(map[cx+cy*size_x] > p2l(0.7) && flag)
            {
                thisLik = BLACK_VALUE;
            }
            likelihood[(cx - x_min) + (cy - y_min) * size_x] = thisLik;
        }
    }

	MRPT_END
}

void COccupancyGridMap2D::precomputedLikelihood_fine6(const int            &x_min,
                                                      const int            &y_min,
                                                      const int            &x_max,
                                                      const int            &y_max,
                                                      std::vector<float>   &likelihood)
{
	MRPT_START
//	int		    K = (int)ceil(LF_maxCorrsDistance_coarse / resolution);	// The size of the checking area for matchings:
	int		    K = FINE_SEARCH_SIZE;	// The size of the checking area for matchings:
	bool		Product_T_OrSum_F = !likelihoodOptions.LF_alternateAverageMethod;

	float		stdHit	= LF_stdHit_fine;
	float		zHit	= likelihoodOptions.LF_zHit;
	float		zRandom	= likelihoodOptions.LF_zRandom;
	float		zRandomMaxRange	= likelihoodOptions.LF_maxRange;
	float		zRandomTerm = zRandom / zRandomMaxRange;
	float		Q = -0.5f / square(stdHit);

	unsigned int	size_x_1 = size_x-1;
	unsigned int	size_y_1 = size_y-1;

	// Aux. variables for the "for j" loop:
	double		thisLik;
	double		maxCorrDist_sq = square(LF_maxCorrsDistance_fine);
	float		occupiedMinDist;

	cellType	thresholdCellValue = p2l(cell_value);

	const double _resolution = this->resolution;
	const double constDist2DiscrUnits = 100 / (_resolution * _resolution);
	const double constDist2DiscrUnits_INV = 1.0 / constDist2DiscrUnits;

	int cx;
	int cy;
	bool flag = true;

    for(cx = x_min; cx < x_max; cx++)
    {
        for(cy = y_min; cy < y_max; cy++)
        {
            flag = true;
            occupiedMinDist = maxCorrDist_sq; // The max.distance

            // Compute now:
            // -------------
            // Find the closest occupied cell in a certain range, given by K:
            int xx1 = max(0,cx-K);
            int xx2 = min(size_x_1,(unsigned)(cx+K));
            int yy1 = max(0,cy-K);
            int yy2 = min(size_y_1,(unsigned)(cy+K));

            // Optimized code: this part will be invoked a *lot* of times:
            {
                cellType  *mapPtr  = &map[xx1+yy1*size_x]; // Initial pointer position
                unsigned   incrAfterRow = size_x - ((xx2-xx1)+1);

                signed int Ax0 = 10*(xx1-cx);
                signed int Ay  = 10*(yy1-cy);

                unsigned int occupiedMinDistInt = mrpt::utils::round( maxCorrDist_sq * constDist2DiscrUnits );

                for (int yy=yy1;yy<=yy2;yy++)
                {
                    unsigned int Ay2 = square((unsigned int)(Ay)); // Square is faster with unsigned.
                    signed short Ax=Ax0;
                    cellType  cell;

                    for (int xx=xx1;xx<=xx2;xx++)
                    {
                        if ( (cell =*mapPtr++) < thresholdCellValue )
                        {
                            unsigned int d = square((unsigned int)(Ax)) + Ay2;
                            keep_min(occupiedMinDistInt, d);
                            flag = false;
                        }
                        Ax += 10;
                    }
                    // Go to (xx1,yy++)
                    mapPtr += incrAfterRow;
                    Ay += 10;
                }

                occupiedMinDist = occupiedMinDistInt * constDist2DiscrUnits_INV ;
            }

            thisLik = zRandomTerm  + zHit * exp( Q * occupiedMinDist );
            if(map[cx+cy*size_x] > p2l(0.7) && flag)
            {
                thisLik = BLACK_VALUE;
            }
            likelihood[(cx - x_min) + (cy - y_min) * size_x] = thisLik;
        }
    }

	MRPT_END
}

void COccupancyGridMap2D::precomputedLikelihood_fine7(const int            &x_min,
                                                      const int            &y_min,
                                                      const int            &x_max,
                                                      const int            &y_max,
                                                      std::vector<float>   &likelihood)
{
	MRPT_START
//	int		    K = (int)ceil(LF_maxCorrsDistance_coarse / resolution);	// The size of the checking area for matchings:
	int		    K = FINE_SEARCH_SIZE;	// The size of the checking area for matchings:
	bool		Product_T_OrSum_F = !likelihoodOptions.LF_alternateAverageMethod;

	float		stdHit	= LF_stdHit_fine;
	float		zHit	= likelihoodOptions.LF_zHit;
	float		zRandom	= likelihoodOptions.LF_zRandom;
	float		zRandomMaxRange	= likelihoodOptions.LF_maxRange;
	float		zRandomTerm = zRandom / zRandomMaxRange;
	float		Q = -0.5f / square(stdHit);

	unsigned int	size_x_1 = size_x-1;
	unsigned int	size_y_1 = size_y-1;

	// Aux. variables for the "for j" loop:
	double		thisLik;
	double		maxCorrDist_sq = square(LF_maxCorrsDistance_fine);
	float		occupiedMinDist;

	cellType	thresholdCellValue = p2l(cell_value);

	const double _resolution = this->resolution;
	const double constDist2DiscrUnits = 100 / (_resolution * _resolution);
	const double constDist2DiscrUnits_INV = 1.0 / constDist2DiscrUnits;

	int cx;
	int cy;
	bool flag = true;

    for(cx = x_min; cx < x_max; cx++)
    {
        for(cy = y_min; cy < y_max; cy++)
        {
            flag = true;
            occupiedMinDist = maxCorrDist_sq; // The max.distance

            // Compute now:
            // -------------
            // Find the closest occupied cell in a certain range, given by K:
            int xx1 = max(0,cx-K);
            int xx2 = min(size_x_1,(unsigned)(cx+K));
            int yy1 = max(0,cy-K);
            int yy2 = min(size_y_1,(unsigned)(cy+K));

            // Optimized code: this part will be invoked a *lot* of times:
            {
                cellType  *mapPtr  = &map[xx1+yy1*size_x]; // Initial pointer position
                unsigned   incrAfterRow = size_x - ((xx2-xx1)+1);

                signed int Ax0 = 10*(xx1-cx);
                signed int Ay  = 10*(yy1-cy);

                unsigned int occupiedMinDistInt = mrpt::utils::round( maxCorrDist_sq * constDist2DiscrUnits );

                for (int yy=yy1;yy<=yy2;yy++)
                {
                    unsigned int Ay2 = square((unsigned int)(Ay)); // Square is faster with unsigned.
                    signed short Ax=Ax0;
                    cellType  cell;

                    for (int xx=xx1;xx<=xx2;xx++)
                    {
                        if ( (cell =*mapPtr++) < thresholdCellValue )
                        {
                            unsigned int d = square((unsigned int)(Ax)) + Ay2;
                            keep_min(occupiedMinDistInt, d);
                            flag = false;
                        }
                        Ax += 10;
                    }
                    // Go to (xx1,yy++)
                    mapPtr += incrAfterRow;
                    Ay += 10;
                }

                occupiedMinDist = occupiedMinDistInt * constDist2DiscrUnits_INV ;
            }

            thisLik = zRandomTerm  + zHit * exp( Q * occupiedMinDist );
            if(map[cx+cy*size_x] > p2l(0.7) && flag)
            {
                thisLik = BLACK_VALUE;
            }
            likelihood[(cx - x_min) + (cy - y_min) * size_x] = thisLik;
        }
    }

	MRPT_END
}

void COccupancyGridMap2D::precomputedLikelihood_fine8(const int            &x_min,
                                                      const int            &y_min,
                                                      const int            &x_max,
                                                      const int            &y_max,
                                                      std::vector<float>   &likelihood)
{
	MRPT_START
//	int		    K = (int)ceil(LF_maxCorrsDistance_coarse / resolution);	// The size of the checking area for matchings:
	int		    K = FINE_SEARCH_SIZE;	// The size of the checking area for matchings:
	bool		Product_T_OrSum_F = !likelihoodOptions.LF_alternateAverageMethod;

	float		stdHit	= LF_stdHit_fine;
	float		zHit	= likelihoodOptions.LF_zHit;
	float		zRandom	= likelihoodOptions.LF_zRandom;
	float		zRandomMaxRange	= likelihoodOptions.LF_maxRange;
	float		zRandomTerm = zRandom / zRandomMaxRange;
	float		Q = -0.5f / square(stdHit);

	unsigned int	size_x_1 = size_x-1;
	unsigned int	size_y_1 = size_y-1;

	// Aux. variables for the "for j" loop:
	double		thisLik;
	double		maxCorrDist_sq = square(LF_maxCorrsDistance_fine);
	float		occupiedMinDist;

	cellType	thresholdCellValue = p2l(cell_value);

	const double _resolution = this->resolution;
	const double constDist2DiscrUnits = 100 / (_resolution * _resolution);
	const double constDist2DiscrUnits_INV = 1.0 / constDist2DiscrUnits;

	int cx;
	int cy;
	bool flag = true;

    for(cx = x_min; cx < x_max; cx++)
    {
        for(cy = y_min; cy < y_max; cy++)
        {
            flag = true;
            occupiedMinDist = maxCorrDist_sq; // The max.distance

            // Compute now:
            // -------------
            // Find the closest occupied cell in a certain range, given by K:
            int xx1 = max(0,cx-K);
            int xx2 = min(size_x_1,(unsigned)(cx+K));
            int yy1 = max(0,cy-K);
            int yy2 = min(size_y_1,(unsigned)(cy+K));

            // Optimized code: this part will be invoked a *lot* of times:
            {
                cellType  *mapPtr  = &map[xx1+yy1*size_x]; // Initial pointer position
                unsigned   incrAfterRow = size_x - ((xx2-xx1)+1);

                signed int Ax0 = 10*(xx1-cx);
                signed int Ay  = 10*(yy1-cy);

                unsigned int occupiedMinDistInt = mrpt::utils::round( maxCorrDist_sq * constDist2DiscrUnits );

                for (int yy=yy1;yy<=yy2;yy++)
                {
                    unsigned int Ay2 = square((unsigned int)(Ay)); // Square is faster with unsigned.
                    signed short Ax=Ax0;
                    cellType  cell;

                    for (int xx=xx1;xx<=xx2;xx++)
                    {
                        if ( (cell =*mapPtr++) < thresholdCellValue )
                        {
                            unsigned int d = square((unsigned int)(Ax)) + Ay2;
                            keep_min(occupiedMinDistInt, d);
                            flag = false;
                        }
                        Ax += 10;
                    }
                    // Go to (xx1,yy++)
                    mapPtr += incrAfterRow;
                    Ay += 10;
                }

                occupiedMinDist = occupiedMinDistInt * constDist2DiscrUnits_INV ;
            }

            thisLik = zRandomTerm  + zHit * exp( Q * occupiedMinDist );
            if(map[cx+cy*size_x] > p2l(0.7) && flag)
            {
                thisLik = BLACK_VALUE;
            }
            likelihood[(cx - x_min) + (cy - y_min) * size_x] = thisLik;
        }
    }

	MRPT_END
}*/
