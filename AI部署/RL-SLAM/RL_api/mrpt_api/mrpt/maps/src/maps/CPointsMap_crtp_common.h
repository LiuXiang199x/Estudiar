/* +---------------------------------------------------------------------------+
   |                     Mobile Robot Programming Toolkit (MRPT)               |
   |                          http://www.mrpt.org/                             |
   |                                                                           |
   | Copyright (c) 2005-2014, Individual contributors, see AUTHORS file        |
   | See: http://www.mrpt.org/Authors - All rights reserved.                   |
   | Released under BSD License. See details in http://www.mrpt.org/License    |
   +---------------------------------------------------------------------------+ */
#ifndef cpointsmap_crtp_common_H
#define cpointsmap_crtp_common_H

#include <mrpt/utils/round.h>
#include <mrpt/slam/CObservation3DRangeScan.h>
#include <mrpt/slam/CObservation2DRangeScan.h>

namespace mrpt
{
namespace slam
{
namespace detail
{

	template <class Derived>
	struct loadFromRangeImpl
	{
		static inline void  templ_loadFromRangeScan(
			Derived &obj,
			const CObservation2DRangeScan		&rangeScan,
			const CPose3D						*robotPose )
		{
			obj.mark_as_modified();

			CPose3D sensorPose3D(UNINITIALIZED_POSE);
			if (!robotPose)
					sensorPose3D = rangeScan.sensorPose;
			else	sensorPose3D.composeFrom(*robotPose, rangeScan.sensorPose);

			const int sizeRangeScan = rangeScan.scan.size();
			if (!sizeRangeScan)
				return; // Nothing to do.

			if ( obj.x.size()+sizeRangeScan > obj.x.capacity() )
			{
				obj.reserve( (size_t) (obj.x.size() * 1.2f) + 3*sizeRangeScan );
			}

			mrpt::slam::CPointsMap::TLaserRange2DInsertContext  lric(rangeScan);
			sensorPose3D.getHomogeneousMatrix(lric.HM);

			// For quicker access as "float" numbers:
			float		m00 = lric.HM.get_unsafe(0,0);
			float		m01 = lric.HM.get_unsafe(0,1);
			float		m03 = lric.HM.get_unsafe(0,3);
			float		m10 = lric.HM.get_unsafe(1,0);
			float		m11 = lric.HM.get_unsafe(1,1);
			float		m13 = lric.HM.get_unsafe(1,3);
			float		m20 = lric.HM.get_unsafe(2,0);
			float		m21 = lric.HM.get_unsafe(2,1);
			float		m23 = lric.HM.get_unsafe(2,3);

			float		lx_1,ly_1,lz_1,lx=0,ly=0,lz=0; // Punto anterior y actual:
			float		lx_2,ly_2;				 // Punto antes del anterior

			// Initial last point:
			lx_1 = -100; ly_1 = -100; lz_1 = -100;
			lx_2 = -100; ly_2 = -100;

			const bool   useMinDist = obj.insertionOptions.minDistBetweenLaserPoints>0;
			const float  minDistSqrBetweenLaserPoints = square( obj.insertionOptions.minDistBetweenLaserPoints );

			bool	lastPointWasValid = true;
			bool	thisIsTheFirst = true;
			bool  	lastPointWasInserted = false;

			pointmap_traits<Derived>::internal_loadFromRangeScan2D_init(obj, lric);

			const size_t nPointsAtStart = obj.size();
			size_t nextPtIdx = nPointsAtStart;
			{
				const size_t expectedMaxSize = nPointsAtStart+(sizeRangeScan* (obj.insertionOptions.also_interpolate ? 3:1) );
				obj.x.resize( expectedMaxSize );
				obj.y.resize( expectedMaxSize );
				obj.z.resize( expectedMaxSize );
			}

			const CSinCosLookUpTableFor2DScans::TSinCosValues & sincos_vals = obj.m_scans_sincos_cache.getSinCosForScan(rangeScan);
			mrpt_Eigen::Array<float,mrpt_Eigen::Dynamic,1>  scan_gx(sizeRangeScan+3), scan_gy(sizeRangeScan+3),scan_gz(sizeRangeScan+3);  // The +3 is to assure there's room for "nPackets*4"
			{
				mrpt_Eigen::Array<float,mrpt_Eigen::Dynamic,1>  scan_x(sizeRangeScan+3), scan_y(sizeRangeScan+3);

				const mrpt_Eigen::Map<mrpt_Eigen::Matrix<float,mrpt_Eigen::Dynamic,1> > scan_vals( const_cast<float*>(&rangeScan.scan[0]),rangeScan.scan.size(),1 );
				const mrpt_Eigen::Map<mrpt_Eigen::Matrix<float,mrpt_Eigen::Dynamic,1> > ccos( const_cast<float*>(&sincos_vals.ccos[0]),rangeScan.scan.size(),1 );
				const mrpt_Eigen::Map<mrpt_Eigen::Matrix<float,mrpt_Eigen::Dynamic,1> > csin( const_cast<float*>(&sincos_vals.csin[0]),rangeScan.scan.size(),1 );

				scan_x = scan_vals.array() * ccos.array();
				scan_y = scan_vals.array() * csin.array();

				scan_gx = m00*scan_x+m01*scan_y+m03;
				scan_gy = m10*scan_x+m11*scan_y+m13;
				scan_gz = m20*scan_x+m21*scan_y+m23;
			}

			float sum_weight = 0.0;
			float count_weight = 0.0;

			for (int i=0;i<sizeRangeScan;i++)
			{
				if ( rangeScan.validRange[i] )
				{
					lx = scan_gx[i];
					ly = scan_gy[i];
					lz = scan_gz[i];

					lastPointWasInserted = false;
					bool pt_pass_min_dist = true;
					float d2 = 0;
					if (useMinDist)
					{
						if (!lastPointWasValid)
								pt_pass_min_dist = false;
						else
						{
							d2 = (square(lx-lx_1) + square(ly-ly_1) + square(lz-lz_1) );
							pt_pass_min_dist = (d2 > minDistSqrBetweenLaserPoints);
						}
					}

					if ( thisIsTheFirst || pt_pass_min_dist )
					{
						thisIsTheFirst = false;

						obj.x[nextPtIdx] = lx;
						obj.y[nextPtIdx] = ly;
						obj.z[nextPtIdx] = lz;

						///current weight ///add by sara
						float current_weight = (1.7454 / 5.0) *  rangeScan.scan[i];
						obj.z[nextPtIdx] = current_weight;
						sum_weight = sum_weight + current_weight;
						count_weight = count_weight + 1.0;
						///

						nextPtIdx++;

						lastPointWasInserted = true;
						if (useMinDist)
						{
							lx_2 = lx_1;
							ly_2 = ly_1;

							lx_1 = lx;
							ly_1 = ly;
							lz_1 = lz;
						}
					}
				}

				lastPointWasValid = rangeScan.validRange[i] != 0;
			}

			// The last point
			if (lastPointWasValid && !lastPointWasInserted)
			{
				if( !obj.m_heightfilter_enabled || (lz >= obj.m_heightfilter_z_min && lz <= obj.m_heightfilter_z_max ) )
				{
					obj.x[nextPtIdx] = lx;
					obj.y[nextPtIdx] = ly;
					obj.z[nextPtIdx] = lz;
					nextPtIdx++;
				}
			}

			obj.x.resize( nextPtIdx );
			obj.y.resize( nextPtIdx );
			obj.z.resize( nextPtIdx );

			///add by sara
			if(sum_weight > 0.01 && count_weight > 0.01)
			{
				for(size_t j = 0; j < obj.x.size(); j++)
				{
					obj.z[j] = obj.z[j] / (sum_weight / count_weight);
				}
			}
			///
		}

		static inline void  templ_loadFromRangeScan(
			Derived &obj,
			const CObservation3DRangeScan		&rangeScan,
			const CPose3D						*robotPose )
		{
			obj.mark_as_modified();

			// If robot pose is supplied, compute sensor pose relative to it.
			CPose3D sensorPose3D(UNINITIALIZED_POSE);
			if (!robotPose)
					sensorPose3D = rangeScan.sensorPose;
			else	sensorPose3D.composeFrom(*robotPose, rangeScan.sensorPose);

			// Insert vs. load and replace:
			if (!obj.insertionOptions.addToExistingPointsMap)
				obj.resize(0); // Resize to 0 instead of clear() so the std::vector<> memory is not actually deadllocated and can be reused.

			if (!rangeScan.hasPoints3D)
				return; // Nothing to do!

			const size_t sizeRangeScan = rangeScan.points3D_x.size();

			// For a great gain in efficiency:
			if ( obj.x.size()+sizeRangeScan> obj.x.capacity() )
				obj.reserve( size_t(obj.x.size() + 1.1*sizeRangeScan) );


			// GENERAL CASE OF SCAN WITH ARBITRARY 3D ORIENTATION:
			// --------------------------------------------------------------------------
			mrpt::slam::CPointsMap::TLaserRange3DInsertContext  lric(rangeScan);
			sensorPose3D.getHomogeneousMatrix(lric.HM);
			// For quicker access to values as "float" instead of "doubles":
			float		m00 = lric.HM.get_unsafe(0,0);
			float		m01 = lric.HM.get_unsafe(0,1);
			float		m02 = lric.HM.get_unsafe(0,2);
			float		m03 = lric.HM.get_unsafe(0,3);
			float		m10 = lric.HM.get_unsafe(1,0);
			float		m11 = lric.HM.get_unsafe(1,1);
			float		m12 = lric.HM.get_unsafe(1,2);
			float		m13 = lric.HM.get_unsafe(1,3);
			float		m20 = lric.HM.get_unsafe(2,0);
			float		m21 = lric.HM.get_unsafe(2,1);
			float		m22 = lric.HM.get_unsafe(2,2);
			float		m23 = lric.HM.get_unsafe(2,3);

			float		lx_1,ly_1,lz_1,lx=0,ly=0,lz=0;		// Punto anterior y actual:

			// Initial last point:
			lx_1 = -100; ly_1 = -100; lz_1 = -100;

			float  minDistSqrBetweenLaserPoints = square( obj.insertionOptions.minDistBetweenLaserPoints );

			// If the user doesn't want a minimum distance:
			if (obj.insertionOptions.minDistBetweenLaserPoints<=0)
				minDistSqrBetweenLaserPoints = -1;

			// ----------------------------------------------------------------
			//   Transform these points into 3D using the pose transformation:
			// ----------------------------------------------------------------
			bool	lastPointWasValid = true;
			bool	thisIsTheFirst = true;
			bool  	lastPointWasInserted = false;

			// Initialize extra stuff in derived class:
			pointmap_traits<Derived>::internal_loadFromRangeScan3D_init(obj,lric);

			for (size_t i=0;i<sizeRangeScan;i++)
			{
				// Valid point?
				if ( rangeScan.points3D_x[i]!=0 || rangeScan.points3D_y[i]!=0 || rangeScan.points3D_z[i]!=0 || obj.insertionOptions.insertInvalidPoints)
				{
					lric.scan_x = rangeScan.points3D_x[i];
					lric.scan_y = rangeScan.points3D_y[i];
					lric.scan_z = rangeScan.points3D_z[i];

					lx = m00*lric.scan_x + m01*lric.scan_y + m02*lric.scan_z + m03;
					ly = m10*lric.scan_x + m11*lric.scan_y + m12*lric.scan_z + m13;
					lz = m20*lric.scan_x + m21*lric.scan_y + m22*lric.scan_z + m23;

					// Specialized work in derived classes:
					pointmap_traits<Derived>::internal_loadFromRangeScan3D_prepareOneRange(obj,lx,ly,lz,lric);

					lastPointWasInserted = false;

					// Add if distance > minimum only:
					float d2 = (square(lx-lx_1) + square(ly-ly_1) + square(lz-lz_1) );
					if ( thisIsTheFirst || (lastPointWasValid && (d2 > minDistSqrBetweenLaserPoints)) )
					{
						thisIsTheFirst = false;

						obj.x.push_back( lx );
						obj.y.push_back( ly );
						obj.z.push_back( lz );
						// Allow derived classes to add any other information to that point:
						pointmap_traits<Derived>::internal_loadFromRangeScan3D_postPushBack(obj,lric);

						lastPointWasInserted = true;

						lx_1 = lx;
						ly_1 = ly;
						lz_1 = lz;
					}

					lastPointWasValid = true;
				}
				else
				{
					lastPointWasValid = false;
				}

				pointmap_traits<Derived>::internal_loadFromRangeScan3D_postOneRange(obj,lric);
			}

			// The last point
			if (lastPointWasValid && !lastPointWasInserted)
			{
				if (lx!=0 || ly!=0 || lz!=0)
				{
					obj.x.push_back( lx );
					obj.y.push_back( ly );
					obj.z.push_back( lz );
					// Allow derived classes to add any other information to that point:
					pointmap_traits<Derived>::internal_loadFromRangeScan3D_postPushBack(obj,lric);
				}
			}
		}

	};

} // end NS
} // end NS
} // end NS

#endif


