/* +---------------------------------------------------------------------------+
   |                     Mobile Robot Programming Toolkit (MRPT)               |
   |                          http://www.mrpt.org/                             |
   |                                                                           |
   | Copyright (c) 2005-2014, Individual contributors, see AUTHORS file        |
   | See: http://www.mrpt.org/Authors - All rights reserved.                   |
   | Released under BSD License. See details in http://www.mrpt.org/License    |
   +---------------------------------------------------------------------------+ */
#ifndef CMonteCarloLocalization2D_H
#define CMonteCarloLocalization2D_H

#include <mrpt/poses/CPosePDFParticles.h>
#include <mrpt/slam/PF_implementations_data.h>
#include <mrpt/slam/TMonteCarloLocalizationParams.h>

#include <mrpt/slam/link_pragmas.h>

namespace mrpt
{
	/** \ingroup mrpt_slam_grp */
	namespace slam
	{
		class COccupancyGridMap2D;
		class CSensoryFrame;

		using namespace mrpt::poses;
		using namespace mrpt::slam;
		using namespace mrpt::bayes;

		/** Declares a class that represents a Probability Density Function (PDF) over a 2D pose (x,y,phi), using a set of weighted samples.
		 *
		 *  This class also implements particle filtering for robot localization. See the MRPT
		 *   application "app/pf-localization" for an example of usage.
		 *
		 * \sa CMonteCarloLocalization3D, CPose2D, CPosePDF, CPoseGaussianPDF, CParticleFilterCapable
		 * \ingroup mrpt_slam_grp
		 */
		class SLAM_IMPEXP CMonteCarloLocalization2D :
			public CPosePDFParticles,
			public PF_implementation<mrpt::poses::CPose2D,CMonteCarloLocalization2D>
		{
		public:
			TMonteCarloLocalizationParams	options; //!< MCL parameters

			/** Constructor
			  * \param M The number of m_particles.
			  */
			CMonteCarloLocalization2D( size_t M = 1 );

			/** Destructor */
			virtual ~CMonteCarloLocalization2D();

			/** Reset the PDF to an uniformly distributed one, but only in the free-space
			  *   of a given 2D occupancy-grid-map. Orientation is randomly generated in the whole 2*PI range.
			  * \param theMap The occupancy grid map
			  * \param freeCellsThreshold The minimum free-probability to consider a cell as empty (default is 0.7)
			  * \param particlesCount If set to -1 the number of m_particles remains unchanged.
			  * \param x_min The limits of the area to look for free cells.
			  * \param x_max The limits of the area to look for free cells.
			  * \param y_min The limits of the area to look for free cells.
			  * \param y_max The limits of the area to look for free cells.
			  * \param phi_min The limits of the area to look for free cells.
			  * \param phi_max The limits of the area to look for free cells.
			  *  \sa resetDeterm32inistic
			  * \exception std::exception On any error (no free cell found in map, map=NULL, etc...)
			  */
			bool  resetUniformFreeSpace(
						COccupancyGridMap2D		*theMap,
                        const double 					freeCellsThreshold = 0.7,
						const int	 					particlesCount = -1,
						const double 					x_min = -1e10f,
						const double 					x_max = 1e10f,
						const double 					y_min = -1e10f,
						const double 					y_max = 1e10f,
						const double 					phi_min = -M_PI,
						const double 					phi_max = M_PI );

            // --yyang--
            void  yy_resetUniformFreeSpace(
                        COccupancyGridMap2D		*theMap,
                        const double 					freeCellsThreshold ,
                        const int	 					particlesCount ,
                        const double 					x_min ,
                        const double 					x_max ,
                        const double 					y_min ,
                        const double 					y_max ,
                        const double 					phi_min,
                        const double 					phi_max);

	   size_t  calculateFreeSpace(COccupancyGridMap2D		        *theMap,
			const double 					down_freeCellsThreshold ,
			const double 					up_freeCellsThreshold ,
			const double 					x_min ,
			const double 					x_max ,
			const double 					y_min ,
			const double 					y_max ,
			const double 					phi_min,
			const double 					phi_max);

            bool  resetDoorUniformFreeSpace(
						COccupancyGridMap2D		*theMap,
                        const double 					down_freeCellsThreshold = 0.7,
                        const double 					up_freeCellsThreshold = 1.0,
						const double 					x_min = -1e10f,
						const double 					x_max = 1e10f,
						const double 					y_min = -1e10f,
						const double 					y_max = 1e10f,
						const double 					phi_min = -M_PI,
						const double 					phi_max = M_PI );

            bool  resetFirstUniformFreeSpace(
						COccupancyGridMap2D		*theMap,
                        const double 					down_freeCellsThreshold = 0.7,
                        const double 					up_freeCellsThreshold = 1.0,
						const double 					x_min = -1e10f,
						const double 					x_max = 1e10f,
						const double 					y_min = -1e10f,
						const double 					y_max = 1e10f,
						const double 					phi_min = -M_PI,
						const double 					phi_max = M_PI );

            bool  resetFirstUniformFreeSpace(
                        COccupancyGridMap2D		        *theMap,
                        const double 					down_freeCellsThreshold ,
                        const double 					up_freeCellsThreshold ,
                        const double 					x_min ,
                        const double 					x_max ,
                        const double 					y_min ,
                        const double 					y_max ,
                        const double 					phi_min,
                        const double 					phi_max,
                        const int                       sample_size);

            bool  resetSecondUniformFreeSpace(
						COccupancyGridMap2D		*theMap,
                        const double 					down_freeCellsThreshold = 0.7,
                        const double 					up_freeCellsThreshold = 1.0,
						const double 					x_min = -1e10f,
						const double 					x_max = 1e10f,
						const double 					y_min = -1e10f,
						const double 					y_max = 1e10f,
						const double 					phi_min = -M_PI,
						const double 					phi_max = M_PI );

            bool  resetPartUniformFreeSpace(
						COccupancyGridMap2D		*theMap,
                        const double 					down_freeCellsThreshold = 0.7,
                        const double 					up_freeCellsThreshold = 1.0,
						const double 					x_min = -1e10f,
						const double 					x_max = 1e10f,
						const double 					y_min = -1e10f,
						const double 					y_max = 1e10f,
						const double 					phi_min = -M_PI,
						const double 					phi_max = M_PI );

            bool isParticleConvergency(COccupancyGridMap2D		*theMap);

            vector<CPose2D>                yy_robot_pose;
            void yy_setRobotPose(vector<CPose2D> robot_pose);

            size_t getParticleSize(){return m_particles.size();}

            void clearParticles()
            {
                MRPT_START
                for (typename CParticleList::iterator it=m_particles.begin();it!=m_particles.end();it++)
                    if (it->d) delete it->d;

                CParticleList temp;
                m_particles.swap(temp);
                MRPT_END
            }

            double getMostLikeParticleWeight();

            mrpt::bayes::CParticleFilterData<CPose2D>::CParticleList getParticleList();

            void setParticleList(mrpt::bayes::CParticleFilterData<CPose2D>::CParticleList &particle_list);

            void setParticleList(std::deque<mrpt::poses::CPosePDFParticles::TLikelihoodParam>  &particles_deque);

			 /** Update the m_particles, predicting the posterior of robot pose and map after a movement command.
			  *  This method has additional configuration parameters in "options".
			  *  Performs the update stage of the RBPF, using the sensed CSensoryFrame:
			  *
			  *   \param action This is a pointer to CActionCollection, containing the pose change the robot has been commanded.
			  *   \param observation This must be a pointer to a CSensoryFrame object, with robot sensed observations.
			  *
			  * \sa options
			  */
			void  prediction_and_update_pfStandardProposal(
				const mrpt::slam::CActionCollection	* action,
				const mrpt::slam::CSensoryFrame		* observation,
				const bayes::CParticleFilter::TParticleFilterOptions &PF_options );

			 /** Update the m_particles, predicting the posterior of robot pose and map after a movement command.
			  *  This method has additional configuration parameters in "options".
			  *  Performs the update stage of the RBPF, using the sensed CSensoryFrame:
			  *
			  *   \param Action This is a pointer to CActionCollection, containing the pose change the robot has been commanded.
			  *   \param observation This must be a pointer to a CSensoryFrame object, with robot sensed observations.
			  *
			  * \sa options
			  */
			void  prediction_and_update_pfAuxiliaryPFStandard(
				const mrpt::slam::CActionCollection	* action,
				const mrpt::slam::CSensoryFrame		* observation,
				const bayes::CParticleFilter::TParticleFilterOptions &PF_options );

			 /** Update the m_particles, predicting the posterior of robot pose and map after a movement command.
			  *  This method has additional configuration parameters in "options".
			  *  Performs the update stage of the RBPF, using the sensed CSensoryFrame:
			  *
			  *   \param Action This is a pointer to CActionCollection, containing the pose change the robot has been commanded.
			  *   \param observation This must be a pointer to a CSensoryFrame object, with robot sensed observations.
			  *
			  * \sa options
			  */
			void  prediction_and_update_pfAuxiliaryPFOptimal(
				const mrpt::slam::CActionCollection	* action,
				const mrpt::slam::CSensoryFrame		* observation,
				const bayes::CParticleFilter::TParticleFilterOptions &PF_options );

		//protected:
			/** \name Virtual methods that the PF_implementations assume exist.
			    @{ */
			/** Return a pointer to the last robot pose in the i'th particle (or NULL if it's a path and it's empty). */
			const TPose3D * getLastPose(const size_t i) const;

			TPose3D getLastPoseEx(const size_t i);

			void PF_SLAM_implementation_custom_update_particle_with_new_pose(
				CParticleDataContent *particleData,
				const TPose3D &newPose) const;

			// We'll redefine this one:
			void PF_SLAM_implementation_replaceByNewParticleSet(
				CParticleList &old_particles,
				const vector<TPose3D>		&newParticles,
				const vector<double>		&newParticlesWeight,
				const vector<size_t>		&newParticlesDerivedFromIdx ) const;

			/** Evaluate the observation likelihood for one particle at a given location */
			double PF_SLAM_computeObservationLikelihoodForParticle(
				const CParticleFilter::TParticleFilterOptions	&PF_options,
				const size_t			particleIndexForMap,
				const CSensoryFrame		&observation,
				const mrpt::poses::CPose3D &x ) const;
			/** @} */

			double PF_SLAM_computeObservationLikelihoodForParticle(
                            const CParticleFilter::TParticleFilterOptions	&PF_options,
                            const size_t			particleIndexForMap,
                            const CSensoryFrame		&observation,
                            const CPose2D			&x,
                            double                  &min_logw) const;

            TPose3D                      m_auxHolder;

            ///sara
            void sortParticlesDeque(const CParticleData &current_particle_data, const size_t &i);

			void setMostLikelyVectorSize(const int  &deque_size);

			const std::deque<TLikelihoodParam> &getMostParticlesDeque(){return m_particles_deque;}

			void clearMostParticlesDeque(){m_particles_deque.clear();}

            std::deque<TLikelihoodParam>    m_particles_deque;
            TLikelihoodParam                m_last_particles;

		}; // End of class def.

	} // End of namespace
} // End of namespace

#endif
