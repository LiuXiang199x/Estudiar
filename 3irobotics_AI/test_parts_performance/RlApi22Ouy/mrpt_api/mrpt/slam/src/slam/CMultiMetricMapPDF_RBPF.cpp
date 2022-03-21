/* +---------------------------------------------------------------------------+
   |                     Mobile Robot Programming Toolkit (MRPT)               |
   |                          http://www.mrpt.org/                             |
   |                                                                           |
   | Copyright (c) 2005-2014, Individual contributors, see AUTHORS file        |
   | See: http://www.mrpt.org/Authors - All rights reserved.                   |
   | Released under BSD License. See details in http://www.mrpt.org/License    |
   +---------------------------------------------------------------------------+ */

#include "slam-precomp.h"   // Precompiled headers

#include <mrpt/random.h>
#include <mrpt/math/utils.h>
#include <mrpt/utils/CTicTac.h>
#include <mrpt/utils/CFileStream.h>

#include <mrpt/slam/CMultiMetricMapPDF.h>
#include <mrpt/slam/CActionRobotMovement2D.h>
#include <mrpt/slam/CActionRobotMovement3D.h>
#include <mrpt/slam/CActionCollection.h>
#include <mrpt/poses/CPosePDFGaussian.h>
#include <mrpt/poses/CPosePDFGrid.h>
#include <mrpt/slam/CSimplePointsMap.h>
#include <mrpt/math.h>

#include <mrpt/slam/PF_aux_structs.h>


using namespace mrpt;
using namespace mrpt::math;
using namespace mrpt::slam;
using namespace mrpt::poses;
using namespace mrpt::random;
using namespace mrpt::utils;
using namespace std;

namespace mrpt
{
	namespace slam
	{
		/** Fills out a "TPoseBin2D" variable, given a path hypotesis and (if not set to NULL) a new pose appended at the end, using the KLD params in "options".
			*/
		template <>
		void KLF_loadBinFromParticle(
			detail::TPoseBin2D	&outBin,
			const TKLDParams  	&opts,
			const CRBPFParticleData	*currentParticleValue,
			const TPose3D			*newPoseToBeInserted)
		{
			// 2D pose approx: Use the latest pose only:
			if (newPoseToBeInserted)
			{
				outBin.x 	= round( newPoseToBeInserted->x / opts.KLD_binSize_XY );
				outBin.y	= round( newPoseToBeInserted->y / opts.KLD_binSize_XY );
				outBin.phi	= round( newPoseToBeInserted->yaw / opts.KLD_binSize_PHI );
			}
			else
			{
				ASSERT_(currentParticleValue && !currentParticleValue->robotPath.empty())
				const TPose3D &p = *currentParticleValue->robotPath.rbegin();
				outBin.x 	= round( p.x / opts.KLD_binSize_XY );
				outBin.y	= round( p.y / opts.KLD_binSize_XY );
				outBin.phi	= round( p.yaw / opts.KLD_binSize_PHI );
			}
		}

		/** Fills out a "TPathBin2D" variable, given a path hypotesis and (if not set to NULL) a new pose appended at the end, using the KLD params in "options".
			*/
		template <>
		void KLF_loadBinFromParticle(
			detail::TPathBin2D	&outBin,
			const TKLDParams  	&opts,
			const CRBPFParticleData	*currentParticleValue,
			const TPose3D			*newPoseToBeInserted)
		{
			const size_t lenBinPath = (currentParticleValue!=NULL) ? currentParticleValue->robotPath.size() : 0;

			// Set the output bin dimensionality:
			outBin.bins.resize(lenBinPath + (newPoseToBeInserted!=NULL ? 1:0) );

			// Is a path provided??
			if (currentParticleValue!=NULL)
				for (size_t i=0;i<lenBinPath;++i)	// Fill the bin data:
				{
					outBin.bins[i].x   = round( currentParticleValue->robotPath[i].x / opts.KLD_binSize_XY );
					outBin.bins[i].y   = round( currentParticleValue->robotPath[i].y / opts.KLD_binSize_XY );
					outBin.bins[i].phi = round( currentParticleValue->robotPath[i].yaw / opts.KLD_binSize_PHI );
				}

			// Is a newPose provided??
			if (newPoseToBeInserted!=NULL)
			{
				// And append the last pose: the new one:
				outBin.bins[lenBinPath].x   = round( newPoseToBeInserted->x / opts.KLD_binSize_XY );
				outBin.bins[lenBinPath].y   = round( newPoseToBeInserted->y / opts.KLD_binSize_XY );
				outBin.bins[lenBinPath].phi = round( newPoseToBeInserted->yaw / opts.KLD_binSize_PHI );
			}
		}
	}
}

// Include this AFTER specializations:
#include <mrpt/slam/PF_implementations.h>

/** Auxiliary for optimal sampling in RO-SLAM */
struct TAuxRangeMeasInfo
{
	TAuxRangeMeasInfo() :
		sensorLocationOnRobot(),
		sensedDistance(0),
		beaconID(INVALID_BEACON_ID),
		nGaussiansInMap(0)
	{}

	CPoint3D		sensorLocationOnRobot;
	float			sensedDistance;
	int64_t			beaconID;
	size_t			nGaussiansInMap; // Number of Gaussian modes in the map representation

	/** Auxiliary for optimal sampling in RO-SLAM */
	static bool cmp_Asc(const TAuxRangeMeasInfo &a, const TAuxRangeMeasInfo &b)
	{
		return a.nGaussiansInMap < b.nGaussiansInMap;
	}
};



/*----------------------------------------------------------------------------------
			prediction_and_update_pfAuxiliaryPFOptimal

  See paper reference in "PF_SLAM_implementation_pfAuxiliaryPFOptimal"
 ----------------------------------------------------------------------------------*/
void  CMultiMetricMapPDF::prediction_and_update_pfAuxiliaryPFOptimal(
	const mrpt::slam::CActionCollection	* actions,
	const mrpt::slam::CSensoryFrame		* sf,
	const bayes::CParticleFilter::TParticleFilterOptions &PF_options )
{
	MRPT_START

	PF_SLAM_implementation_pfAuxiliaryPFOptimal<mrpt::slam::detail::TPoseBin2D>( actions, sf, PF_options,options.KLD_params);

	MRPT_END
}



/*----------------------------------------------------------------------------------
			prediction_and_update_pfOptimalProposal

For grid-maps:
==============
 Approximation by Grissetti et al:  Use scan matching to approximate
   the observation model by a Gaussian:
  See: "Improved Grid-based SLAM with Rao-Blackwellized PF by Adaptive Proposals
	       and Selective Resampling" (G. Grisetti, C. Stachniss, W. Burgard)

For beacon maps:
===============
  (JLBC: Method under development)

 ----------------------------------------------------------------------------------*/
void  CMultiMetricMapPDF::prediction_and_update_pfOptimalProposal(
	const mrpt::slam::CActionCollection	* actions,
	const mrpt::slam::CSensoryFrame		* sf,
	const bayes::CParticleFilter::TParticleFilterOptions &PF_options )
{
	MRPT_START

	// ----------------------------------------------------------------------
	//						PREDICTION STAGE
	// ----------------------------------------------------------------------
	CVectorDouble				rndSamples;
	size_t						M = m_particles.size();
	bool						updateStageAlreadyDone = false;
	CPose3D						initialPose,incrPose, finalPose;

	// ICP used if "pfOptimalProposal_mapSelection" = 0 or 1
	CICP				icp (options.icp_params);  // Set our ICP params instead of default ones.
	CICP::TReturnInfo	icpInfo;

	CParticleList::iterator		partIt;

	ASSERT_(sf!=NULL)

	// Find a robot movement estimation:
	CPose3D						motionModelMeanIncr;	// The mean motion increment:
	CPoseRandomSampler			robotActionSampler;
	{
		CActionRobotMovement2DPtr	robotMovement2D = actions->getBestMovementEstimation();

		// If there is no 2D action, look for a 3D action:
		if (robotMovement2D.present())
		{
			robotActionSampler.setPosePDF( robotMovement2D->poseChange );
			motionModelMeanIncr = robotMovement2D->poseChange->getMeanVal();
		}
		else
		{
			CActionRobotMovement3DPtr	robotMovement3D = actions->getActionByClass<CActionRobotMovement3D>();
			if (robotMovement3D)
			{
				robotActionSampler.setPosePDF( robotMovement3D->poseChange );
				robotMovement3D->poseChange.getMean( motionModelMeanIncr );
			}
			else
			{
				motionModelMeanIncr.setFromValues(0,0,0);
			}
		}
	}

	// Average map will need to be updated after this:
	averageMapIsUpdated = false;

	// --------------------------------------------------------------------------------------
	//  Prediction:
	//
	//  Compute a new mean and covariance by sampling around the mean of the input "action"
	// --------------------------------------------------------------------------------------
	printf(" 1) Prediction...");
	M = m_particles.size();

	// To be computed as an average from all m_particles:
	size_t particleWithHighestW = 0;
	for (size_t i=0;i<M;i++)
		if (getW(i)>getW(particleWithHighestW))
			particleWithHighestW = i;


	//   The paths MUST already contain the starting location for each particle:
	ASSERT_( !m_particles[0].d->robotPath.empty() )

	// Build the local map of points for ICP:
	CSimplePointsMap	localMapPoints;

	bool built_map_points = false;

	// Update particle poses:
	size_t i;
	for (i=0,partIt = m_particles.begin(); partIt!=m_particles.end(); partIt++,i++)
	{
		double extra_log_lik = 0; // Used for the optimal_PF with ICP

		// Set initial robot pose estimation for this particle:
		const CPose3D ith_last_pose = CPose3D(*partIt->d->robotPath.rbegin()); // The last robot pose in the path

		CPose3D		initialPoseEstimation = ith_last_pose + motionModelMeanIncr;

		// Use ICP with the map associated to particle?
		if ( options.pfOptimalProposal_mapSelection==0 ||
			 options.pfOptimalProposal_mapSelection==1 ||
			 options.pfOptimalProposal_mapSelection==3 )
		{
			CPosePDFGaussian			icpEstimation;

			// Configure the matchings that will take place in the ICP process:
			if (partIt->d->mapTillNow.m_pointsMaps.size())
			{
				ASSERT_(partIt->d->mapTillNow.m_pointsMaps.size()==1);
				//partIt->d->mapTillNow.m_pointsMaps[0]->insertionOptions.matchStaticPointsOnly = false;
			}

			CMetricMap *map_to_align_to = NULL;

			if (options.pfOptimalProposal_mapSelection==0)  // Grid map
			{
				ASSERT_( !partIt->d->mapTillNow.m_gridMaps.empty() );

				// Build local map of points.
				if (!built_map_points)
				{
					built_map_points=true;

					localMapPoints.insertionOptions.minDistBetweenLaserPoints =  0.02f; //3.0f * m_particles[0].d->mapTillNow.m_gridMaps[0]->getResolution();;
					localMapPoints.insertionOptions.isPlanarMap = true;
					sf->insertObservationsInto( &localMapPoints );
				}

				map_to_align_to = partIt->d->mapTillNow.m_gridMaps[0].pointer();
			}
			else
			if (options.pfOptimalProposal_mapSelection==3)  // Map of points
			{
				ASSERT_( !partIt->d->mapTillNow.m_pointsMaps.empty() );

				// Build local map of points.
				if (!built_map_points)
				{
					built_map_points=true;

					localMapPoints.insertionOptions.minDistBetweenLaserPoints =  0.02f; //3.0f * m_particles[0].d->mapTillNow.m_gridMaps[0]->getResolution();;
					localMapPoints.insertionOptions.isPlanarMap = true;
					sf->insertObservationsInto( &localMapPoints );
				}

				map_to_align_to = partIt->d->mapTillNow.m_pointsMaps[0].pointer();
			}
			else
			{

			}

			ASSERT_(map_to_align_to!=NULL);

			// Add noise to each particle
			CMatrixDouble cov_initialize;
			robotActionSampler.getOriginalPDFCov2D(cov_initialize);
            CPose3D pose_noise = CPose3D(initialPoseEstimation);

            // Add to the new robot pose:
			randomGenerator.drawGaussianMultivariate(rndSamples,cov_initialize);

			// Add noise:
			pose_noise.setFromValues(
				pose_noise.x() + rndSamples[0]*5.0,
				pose_noise.y() + rndSamples[1]*5.0,
				pose_noise.z(),
				pose_noise.yaw() + rndSamples[2]*5.0,
				pose_noise.pitch(),
				pose_noise.roll() );

			// Use ICP to align to each particle's map:
			{
				CPosePDFPtr alignEst =
				icp.Align(
					map_to_align_to,
					&localMapPoints,
					CPose2D(initialPoseEstimation),
					NULL,
					&icpInfo);
				icpEstimation.copyFrom( *alignEst );
			}


			if (i==particleWithHighestW)
			{
				newInfoIndex = 1 - icpInfo.goodness; //newStaticPointsRatio; //* icpInfo.goodness;
			}


			//printf("[rbpf-slam] gridICP[%u]: %.02f%%\n", i, 100*icpInfo.goodness);
			if (icpInfo.goodness<options.ICPGlobalAlign_MinQuality && SFs.size())
			{
				printf("[rbpf-slam] Warning: gridICP[%u]: %.02f%% -> Using odometry instead!\n", (unsigned int)i, 100*icpInfo.goodness);
				icpEstimation.mean = CPose2D(initialPoseEstimation);
			}

			// Set the gaussian pose:
			CPose3DPDFGaussian finalEstimatedPoseGauss( icpEstimation );

			// As a way to take into account the odometry / "prior", use
			//  a correcting factor in the likelihood from the mismatch prior<->icp_estimate:
//			const double prior_dist_lin = initialPoseEstimation.distanceTo(icpEstimation.mean);
//			const double prior_dist_ang = std::abs( mrpt::math::wrapToPi( initialPoseEstimation.yaw()-icpEstimation.mean.phi() ) );
////			if (prior_dist_lin>0.10 || prior_dist_ang>DEG2RAD(3))
////				printf(" >>>>>>>>>> %f %f\n",prior_dist_lin,RAD2DEG(prior_dist_ang));
//			extra_log_lik = -(prior_dist_lin/0.20) - (prior_dist_ang/DEG2RAD(20));

//				printf("gICP: %.02f%%, Iters=%u\n",icpInfo.goodness,icpInfo.nIterations);

#if 0  // Use hacked ICP covariance:
			CPose3D Ap = finalEstimatedPoseGauss.mean - ith_last_pose;
			const double  Ap_dist = Ap.norm();

			finalEstimatedPoseGauss.cov.zeros();
			finalEstimatedPoseGauss.cov(0,0) = square( fabs(Ap_dist)*0.01 );
			finalEstimatedPoseGauss.cov(1,1) = square( fabs(Ap_dist)*0.01 );
			finalEstimatedPoseGauss.cov(2,2) = square( fabs(Ap.yaw())*0.02 );
#else
		// Use real ICP covariance (with a minimum level):
		keep_max( finalEstimatedPoseGauss.cov(0,0), square(0.002));
		keep_max( finalEstimatedPoseGauss.cov(1,1), square(0.002));
		keep_max( finalEstimatedPoseGauss.cov(2,2), square(DEG2RAD(0.1)));

#endif

			// Generate gaussian-distributed 2D-pose increments according to "finalEstimatedPoseGauss":
			// -------------------------------------------------------------------------------------------
			finalPose = finalEstimatedPoseGauss.mean;					// Add to the new robot pose:
//			randomGenerator.drawGaussianMultivariate(rndSamples, finalEstimatedPoseGauss.cov );
//			// Add noise:
//			finalPose.setFromValues(
//				finalPose.x() + rndSamples[0],
//				finalPose.y() + rndSamples[1],
//				finalPose.z(),
//				finalPose.yaw() + rndSamples[2],
//				finalPose.pitch(),
//				finalPose.roll() );
		}
		else
		{
			// By default:
			// Generate gaussian-distributed 2D-pose increments according to mean-cov:
			if ( !robotActionSampler.isPrepared() )
				THROW_EXCEPTION("Action list does not contain any CActionRobotMovement2D or CActionRobotMovement3D object!");

			robotActionSampler.drawSample( incrPose );

			finalPose = ith_last_pose + incrPose;
		}

		// Insert as the new pose in the path:
		partIt->d->robotPath.push_back( finalPose );

		// ----------------------------------------------------------------------
		//						UPDATE STAGE
		// ----------------------------------------------------------------------
		if (!updateStageAlreadyDone)
		{
			partIt->log_w +=
				PF_options.powFactor *
				(PF_SLAM_computeObservationLikelihoodForParticle(PF_options,i,*sf,finalPose)
				+ extra_log_lik);
		} // if update not already done...

	} // end of for each particle "i" & "partIt"

	printf("Ok\n");

	MRPT_END
}

/*---------------------------------------------------------------
			prediction_and_update_pfStandardProposal
 ---------------------------------------------------------------*/
void  CMultiMetricMapPDF::prediction_and_update_pfStandardProposal(
	const mrpt::slam::CActionCollection	* actions,
	const mrpt::slam::CSensoryFrame		* sf,
	const bayes::CParticleFilter::TParticleFilterOptions &PF_options )
{
	MRPT_START

	PF_SLAM_implementation_pfStandardProposal<mrpt::slam::detail::TPoseBin2D>(actions, sf, PF_options,options.KLD_params);

	// Average map will need to be updated after this:
	averageMapIsUpdated = false;

	MRPT_END
}


// Specialization for my kind of particles:
void CMultiMetricMapPDF::PF_SLAM_implementation_custom_update_particle_with_new_pose(
	CRBPFParticleData *particleData,
	const TPose3D &newPose) const
{
	particleData->robotPath.push_back( newPose );
}

// Specialization for RBPF maps:
bool CMultiMetricMapPDF::PF_SLAM_implementation_doWeHaveValidObservations(
	const CMultiMetricMapPDF::CParticleList	&particles,
	const CSensoryFrame *sf) const
{
	if (sf==NULL) return false;
	ASSERT_(!particles.empty())
	return particles.begin()->d->mapTillNow.canComputeObservationsLikelihood( *sf );
}

/** Do not move the particles until the map is populated.  */
bool CMultiMetricMapPDF::PF_SLAM_implementation_skipRobotMovement() const
{
	return 0==getNumberOfObservationsInSimplemap();
}



/*---------------------------------------------------------------
 Evaluate the observation likelihood for one
   particle at a given location
 ---------------------------------------------------------------*/
double CMultiMetricMapPDF::PF_SLAM_computeObservationLikelihoodForParticle(
	const CParticleFilter::TParticleFilterOptions	&PF_options,
	const size_t			particleIndexForMap,
	const CSensoryFrame		&observation,
	const CPose3D			&x ) const
{
	CMultiMetricMap	*map = &m_particles[particleIndexForMap].d->mapTillNow;
	double	ret = 0;
	for (CSensoryFrame::const_iterator it=observation.begin();it!=observation.end();++it)
		ret += map->computeObservationLikelihood( (CObservation*)it->pointer(), x );
	return ret;
}

