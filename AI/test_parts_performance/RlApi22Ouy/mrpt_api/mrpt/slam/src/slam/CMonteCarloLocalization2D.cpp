/* +---------------------------------------------------------------------------+
   |                     Mobile Robot Programming Toolkit (MRPT)               |
   |                          http://www.mrpt.org/                             |
   |                                                                           |
   | Copyright (c) 2005-2014, Individual contributors, see AUTHORS file        |
   | See: http://www.mrpt.org/Authors - All rights reserved.                   |
   | Released under BSD License. See details in http://www.mrpt.org/License    |
   +---------------------------------------------------------------------------+ */

#include "slam-precomp.h"   // Precompiled headerss

#include <mrpt/slam/CMonteCarloLocalization2D.h>

#include <mrpt/utils/CTicTac.h>
#include <mrpt/slam/COccupancyGridMap2D.h>
#include <mrpt/slam/CActionCollection.h>
#include <mrpt/slam/CSensoryFrame.h>
#include <mrpt/poses/CPose2D.h>

#include <mrpt/random.h>

#include <mrpt/slam/PF_aux_structs.h>

using namespace mrpt;
using namespace mrpt::bayes;
using namespace mrpt::poses;
using namespace mrpt::math;
using namespace mrpt::slam;
using namespace mrpt::random;
using namespace mrpt::utils;
using namespace std;

#include <mrpt/slam/PF_implementations_data.h>

#define debug 1

namespace mrpt
{
	namespace slam
	{
		/** Fills out a "TPoseBin2D" variable, given a path hypotesis and (if not set to NULL) a new pose appended at the end, using the KLD params in "options". */
		template <>
		void KLF_loadBinFromParticle(
			mrpt::slam::detail::TPoseBin2D &outBin,
			const TKLDParams  	&opts,
			const CMonteCarloLocalization2D::CParticleDataContent 	*currentParticleValue,
			const TPose3D		*newPoseToBeInserted)
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
				ASSERT_(currentParticleValue)
				outBin.x 	= round( currentParticleValue->x() / opts.KLD_binSize_XY );
				outBin.y	= round( currentParticleValue->y() / opts.KLD_binSize_XY );
				outBin.phi	= round( currentParticleValue->phi() / opts.KLD_binSize_PHI );
			}
		}
	}
}

#include <mrpt/slam/PF_implementations.h>

//#if defined(_MSC_VER)
//#	pragma warning(push)
//#	pragma warning(disable:4355) // for the "this" argument below
//#endif

/*---------------------------------------------------------------
				ctor
 ---------------------------------------------------------------*/
// Passing a "this" pointer at this moment is not a problem since it will be NOT access until the object is fully initialized
CMonteCarloLocalization2D::CMonteCarloLocalization2D( size_t M ) :
	CPosePDFParticles(M)
//	PF_implementation<CPose2D>(static_cast<mrpt::bayes::CParticleFilterData<CPose2D>&>(*this),static_cast<mrpt::bayes::CParticleFilterCapable&>(*this) )
{
    m_last_particles.particle_pose = CPose2D(-100.0, -100.0, 0.0);
    m_last_particles.particle_logw = -1e300;
}

//#if defined(_MSC_VER)
//#	pragma warning(pop)
//#endif

/*---------------------------------------------------------------
				Dtor
 ---------------------------------------------------------------*/
CMonteCarloLocalization2D::~CMonteCarloLocalization2D()
{
}


/*---------------------------------------------------------------
						getLastPose
 ---------------------------------------------------------------*/
const TPose3D* CMonteCarloLocalization2D::getLastPose(const size_t i) const
{
	if (i>=m_particles.size()) THROW_EXCEPTION("Particle index out of bounds!");
	static TPose3D auxHolder;
	ASSERTDEB_(m_particles[i].d!=NULL)
	auxHolder = TPose3D( TPose2D(*m_particles[i].d));
	return &auxHolder;
}

/*---------------------------------------------------------------
						getLastPose
 ---------------------------------------------------------------*/
TPose3D CMonteCarloLocalization2D::getLastPoseEx(const size_t i)
{
	if (i>=m_particles.size()) THROW_EXCEPTION("Particle index out of bounds!");
//    TPose3D auxHolder;
	ASSERTDEB_(m_particles[i].d!=NULL)
	m_auxHolder = TPose3D( TPose2D(*m_particles[i].d));
	return m_auxHolder;
}

/*---------------------------------------------------------------

			getParticleList

 ---------------------------------------------------------------*/
mrpt::bayes::CParticleFilterData<CPose2D>::CParticleList CMonteCarloLocalization2D::getParticleList()
{
    mrpt::bayes::CParticleFilterData<CPose2D>::CParticleList parts;
    parts.resize(m_particles.size());

    for(size_t i = 0; i < m_particles.size(); i++)
    {
        parts[i].log_w = m_particles[i].log_w;
        parts[i].d = new CParticleFilterData<CPose2D>::CParticleDataContent( *m_particles[i].d );
    }
    return parts;
}

/*---------------------------------------------------------------

			setParticleList

 ---------------------------------------------------------------*/
void CMonteCarloLocalization2D::setParticleList(mrpt::bayes::CParticleFilterData<CPose2D>::CParticleList &particle_list)
{
    clearParticles();

    m_particles.resize(particle_list.size());
    for(size_t i = 0; i < particle_list.size(); i++)
    {
        m_particles[i].log_w = particle_list[i].log_w;
        m_particles[i].d = particle_list[i].d;
    }
    particle_list.clear();
}

/*---------------------------------------------------------------

			setParticleList

 ---------------------------------------------------------------*/
void CMonteCarloLocalization2D::setParticleList(std::deque<mrpt::poses::CPosePDFParticles::TLikelihoodParam>  &particles_deque)
{
    clearParticles();

    m_particles.resize(particles_deque.size());
    for(size_t i = 0; i < particles_deque.size(); i++)
    {
        m_particles[i].log_w = particles_deque[i].particle_logw;
        m_particles[i].d = new CPose2D();
        *(m_particles[i].d) = particles_deque[i].particle_pose;
    }
    particles_deque.clear();
}

/*---------------------------------------------------------------

			prediction_and_update_pfStandardProposal

 ---------------------------------------------------------------*/
void  CMonteCarloLocalization2D::prediction_and_update_pfStandardProposal(
	const mrpt::slam::CActionCollection	* actions,
	const mrpt::slam::CSensoryFrame		* sf,
	const bayes::CParticleFilter::TParticleFilterOptions &PF_options )
{
	MRPT_START

	if (sf)
	{	// A map MUST be supplied!
		ASSERT_(options.metricMap || options.metricMaps.size()>0)
		if (!options.metricMap)
			ASSERT_(options.metricMaps.size() == m_particles.size() )
	}

    if (sf)
    {
        const size_t M = m_particles.size();

        for (size_t i=0;i<M;i++)
        {
            double current_logw = m_particles_deque.back().particle_logw - m_particles[i].log_w;

            const double obs_log_likelihood = PF_SLAM_computeObservationLikelihoodForParticle(PF_options, i, *sf, *m_particles[i].d, current_logw);
            m_particles[i].log_w = m_particles[i].log_w + obs_log_likelihood;

            if(obs_log_likelihood > -1e100)
            {
                sortParticlesDeque(m_particles[i], i);
            }
        }
    }

	MRPT_END
}

void CMonteCarloLocalization2D::sortParticlesDeque(const CParticleData &current_particle_data, const size_t &i)
{
    int particles_size = m_particles.size();
    if(particles_size < 1)
    {
        return;
    }

    int deque_size = m_particles_deque.size();

    double x1 = m_last_particles.particle_pose.m_coords[0];
    double y1 = m_last_particles.particle_pose.m_coords[1];
    double x2 = current_particle_data.d->m_coords[0];
    double y2 = current_particle_data.d->m_coords[1];
    double distance = sqrt(square(x1 - x2) + square(y1 - y2));

    if(distance < 0.01)
    {
        if(current_particle_data.log_w > m_last_particles.particle_logw)
        {
            m_last_particles.particle_pose = *current_particle_data.d;
            m_last_particles.particle_logw = current_particle_data.log_w;
        }
    }
    else
    {
        if(m_last_particles.particle_logw > m_particles_deque.back().particle_logw)
        {
            for(size_t k = 0; k < deque_size; k++)
            {
                if(m_last_particles.particle_logw > m_particles_deque.at(k).particle_logw)
                {
                    for(int j = (deque_size - 1); j >= (k + 1); j--)
                    {
                        m_particles_deque.at(j) = m_particles_deque.at(j - 1);
                    }

                    m_particles_deque.at(k).particle_pose = m_last_particles.particle_pose;
                    m_particles_deque.at(k).particle_logw = m_last_particles.particle_logw;

                    break;
                }
            }
        }
        m_last_particles.particle_pose = *current_particle_data.d;
        m_last_particles.particle_logw = current_particle_data.log_w;
    }

    if(i == (particles_size - 1))
    {
        if(m_last_particles.particle_logw > m_particles_deque.back().particle_logw)
        {
            for(size_t k = 0; k < deque_size; k++)
            {
                if(m_last_particles.particle_logw > m_particles_deque.at(k).particle_logw)
                {
                    for(int j = (deque_size - 1); j >= (k + 1); j--)
                    {
                        m_particles_deque.at(j) = m_particles_deque.at(j - 1);
                    }

                    m_particles_deque.at(k).particle_pose = m_last_particles.particle_pose;
                    m_particles_deque.at(k).particle_logw = m_last_particles.particle_logw;

                    break;
                }
            }
        }
    }
}

/*---------------------------------------------------------------

			prediction_and_update_pfAuxiliaryPFStandard

 ---------------------------------------------------------------*/
void  CMonteCarloLocalization2D::prediction_and_update_pfAuxiliaryPFStandard(
	const mrpt::slam::CActionCollection	* actions,
	const mrpt::slam::CSensoryFrame		* sf,
	const bayes::CParticleFilter::TParticleFilterOptions &PF_options )
{
	MRPT_START

	if (sf)
	{	// A map MUST be supplied!
		ASSERT_(options.metricMap || options.metricMaps.size()>0)
		if (!options.metricMap)
			ASSERT_(options.metricMaps.size() == m_particles.size() )
	}

	PF_SLAM_implementation_pfAuxiliaryPFStandard<mrpt::slam::detail::TPoseBin2D>(actions, sf, PF_options,options.KLD_params);

	MRPT_END
}


/*---------------------------------------------------------------

			prediction_and_update_pfAuxiliaryPFOptimal

 ---------------------------------------------------------------*/
void  CMonteCarloLocalization2D::prediction_and_update_pfAuxiliaryPFOptimal(
	const mrpt::slam::CActionCollection	* actions,
	const mrpt::slam::CSensoryFrame		* sf,
	const bayes::CParticleFilter::TParticleFilterOptions &PF_options )
{
	MRPT_START

	if (sf)
	{	// A map MUST be supplied!
		ASSERT_(options.metricMap || options.metricMaps.size()>0)
		if (!options.metricMap)
			ASSERT_(options.metricMaps.size() == m_particles.size() )
	}

	PF_SLAM_implementation_pfAuxiliaryPFOptimal<mrpt::slam::detail::TPoseBin2D>(actions, sf, PF_options,options.KLD_params);

	MRPT_END
}


/*---------------------------------------------------------------
			PF_SLAM_computeObservationLikelihoodForParticle
 ---------------------------------------------------------------*/
double CMonteCarloLocalization2D::PF_SLAM_computeObservationLikelihoodForParticle(
	const CParticleFilter::TParticleFilterOptions	&PF_options,
	const size_t			particleIndexForMap,
	const CSensoryFrame		&observation,
	const CPose3D			&x ) const
{
	ASSERT_( options.metricMap || particleIndexForMap<options.metricMaps.size() )

	CMetricMap *map = (options.metricMap) ?
		options.metricMap :  // All particles, one map
		options.metricMaps[particleIndexForMap]; // One map per particle

	// For each observation:
	double ret = 0;
	for (CSensoryFrame::const_iterator it=observation.begin();it!=observation.end();++it)
		ret += map->computeObservationLikelihood( it->pointer(), x );	// Compute the likelihood:

	// Done!
	return ret;
}

double CMonteCarloLocalization2D::PF_SLAM_computeObservationLikelihoodForParticle(
	const CParticleFilter::TParticleFilterOptions	&PF_options,
	const size_t			particleIndexForMap,
	const CSensoryFrame		&observation,
	const CPose2D			&x,
	double                  &min_logw) const
{
	CMetricMap *map = options.metricMap;

	double ret = 0;
	for (CSensoryFrame::const_iterator it=observation.begin();it!=observation.end();++it)
		ret += map->computeObservationLikelihood( it->pointer(), x, min_logw);

	return ret;
}

// Specialization for my kind of particles:
void CMonteCarloLocalization2D::PF_SLAM_implementation_custom_update_particle_with_new_pose(
	CPose2D *particleData,
	const TPose3D &newPose) const
{
	*particleData = CPose2D( TPose2D(newPose) );
}


void CMonteCarloLocalization2D::PF_SLAM_implementation_replaceByNewParticleSet(
	CParticleList &old_particles,
	const vector<TPose3D>		&newParticles,
	const vector<double>		&newParticlesWeight,
	const vector<size_t>		&newParticlesDerivedFromIdx ) const
{
	ASSERT_EQUAL_(size_t(newParticlesWeight.size()),size_t(newParticles.size()))

	// ---------------------------------------------------------------------------------
	// Substitute old by new particle set:
	//   Old are in "m_particles"
	//   New are in "newParticles", "newParticlesWeight","newParticlesDerivedFromIdx"
	// ---------------------------------------------------------------------------------
	// Free old m_particles:
	for (size_t i=0;i<old_particles.size();i++)
			mrpt::utils::delete_safe( old_particles[ i ].d );

	// Copy into "m_particles"
	const size_t N = newParticles.size();
	old_particles.resize(N);
	for (size_t i=0;i<N;i++)
	{
		old_particles[i].log_w = newParticlesWeight[i];
		old_particles[i].d = new CPose2D( TPose2D( newParticles[i] ));
	}
}

bool  CMonteCarloLocalization2D::resetUniformFreeSpace(
	COccupancyGridMap2D		*theMap,
	const double 					freeCellsThreshold ,
	const int	 					particlesCount ,
	const double 					x_min ,
	const double 					x_max ,
	const double 					y_min ,
	const double 					y_max ,
	const double 					phi_min,
	const double 					phi_max)
{
	MRPT_START

	ASSERT_(theMap!=NULL)

	int					sizeX = theMap->getSizeX();
	int					sizeY = theMap->getSizeY();
	double				gridRes = theMap->getResolution();
	std::vector<double>	freeCells_x,freeCells_y;
	size_t				nFreeCells;
	unsigned int		xIdx1,xIdx2;
	unsigned int		yIdx1,yIdx2;

	freeCells_x.reserve( sizeX * sizeY );
	freeCells_y.reserve( sizeX * sizeY );

	if (x_min>theMap->getXMin())
			xIdx1 = max(0, theMap->x2idx( x_min ) );
	else	xIdx1 = 0;
	if (x_max<theMap->getXMax())
			xIdx2 = min(sizeX-1, theMap->x2idx( x_max ) );
	else	xIdx2 = sizeX-1;
	if (y_min>theMap->getYMin())
			yIdx1 = max(0, theMap->y2idx( y_min ) );
	else	yIdx1 = 0;
	if (y_max<theMap->getYMax())
			yIdx2 = min(sizeY-1, theMap->y2idx( y_max ) );
	else	yIdx2 = sizeY-1;


	for (unsigned int x=xIdx1;x<=xIdx2;x++)
		for (unsigned int y=yIdx1;y<=yIdx2;y++)
			if (theMap->getCell(x,y)>= freeCellsThreshold)
			{
				freeCells_x.push_back(theMap->idx2x(x));
				freeCells_y.push_back(theMap->idx2y(y));
			}

	nFreeCells = freeCells_x.size();

	// Assure that map is not fully occupied!
	if( nFreeCells )
	{
	    if (particlesCount>0)
        {
            clear();
            m_particles.resize(4 * particlesCount);
            for (int i=0;i<particlesCount;i++)
                m_particles[i].d = new CPose2D();
        }

        const size_t M = m_particles.size();

        // Generate pose m_particles:
        for (size_t i=0;i<M;i++)
        {
            int idx = round(randomGenerator.drawUniform(0.0,nFreeCells-1.001));

            float x = freeCells_x[idx] + randomGenerator.drawUniform( -gridRes, gridRes);
            float y = freeCells_y[idx] + randomGenerator.drawUniform( -gridRes, gridRes);

            for(size_t j = 0; j < 4; j++)
            {
                m_particles[4 * i + j].d->x(x);
                m_particles[4 * i + j].d->y(y);
                m_particles[4 * i + j].d->phi(randomGenerator.drawUniform(phi_min, phi_max));
                m_particles[4 * i + j].log_w = 0;
            }
        }
        return true;
	}
	else
	{
	    return false;
	}

	MRPT_END
}

//----yyang---
void  CMonteCarloLocalization2D::yy_resetUniformFreeSpace(
	COccupancyGridMap2D		*theMap,
	const double 					freeCellsThreshold ,
	const int	 					particlesCount ,
	const double 					x_min ,
	const double 					x_max ,
	const double 					y_min ,
	const double 					y_max ,
	const double 					phi_min,
	const double 					phi_max)
{
	MRPT_START

	double				gridRes = 0.2;
	vector<CPose2D>     robot_pose_list;

	ASSERT_(theMap!=NULL)

	if (particlesCount>0)
	{
		clear();
		m_particles.resize(particlesCount);
		for (int i=0;i<particlesCount;i++)
			m_particles[i].d = new CPose2D();
	}

	for(size_t i = 0; i < yy_robot_pose.size(); ++i)
	{
	    int idxx = theMap->x2idx(yy_robot_pose[i].x());
	    int idxy = theMap->y2idx(yy_robot_pose[i].y());
	    if(theMap->getCell(idxx, idxy) > freeCellsThreshold)
	    {
	        robot_pose_list.push_back(yy_robot_pose[i]);
	    }
	}

	const size_t M = m_particles.size();
	const size_t N = robot_pose_list.size();

    if(N == 1)
	{
	    m_particles[0].d->x( robot_pose_list[0].x() );
        m_particles[0].d->y( robot_pose_list[0].y()  );
        m_particles[0].d->phi( robot_pose_list[0].phi());
        m_particles[0].log_w=0;

        for(size_t i = 1; i < M; i++)
        {
            m_particles[i].d->x( robot_pose_list[0].x() + randomGenerator.drawUniform( -gridRes, gridRes ) );
            m_particles[i].d->y( robot_pose_list[0].y() + randomGenerator.drawUniform( -gridRes, gridRes ) );
            m_particles[i].d->phi( randomGenerator.drawUniform( phi_min, phi_max ) );
            m_particles[i].log_w=0;
        }
	}
	else
	{
	    for(vector<CPose2D>::size_type ix = 0; ix != N; ++ix)
        {
            for (size_t i=size_t(ix*M/N);i<size_t((ix+1)*M/N);i++)
            {
                m_particles[i].d->x( robot_pose_list[ix].x() + randomGenerator.drawUniform( -gridRes, gridRes ) );
                m_particles[i].d->y( robot_pose_list[ix].y() + randomGenerator.drawUniform( -gridRes, gridRes ) );
                m_particles[i].d->phi( randomGenerator.drawUniform( phi_min, phi_max ) );
                m_particles[i].log_w=0;
            }
        }
	}

	MRPT_END
}

void CMonteCarloLocalization2D::yy_setRobotPose(vector<CPose2D> robot_pose)
{
    yy_robot_pose = robot_pose;
}

bool big2small(const pair<int, int> &i, const pair<int, int> &j) {return (i.second > j.second);}

bool CMonteCarloLocalization2D::isParticleConvergency(COccupancyGridMap2D   *theMap)
{
    map<int, int> particle_counter; //key: particle coordinate, value: particle count
    COccupancyGridMap2D x2idx_map(theMap->getXMin(), theMap->getXMax(), theMap->getYMin(),
                                  theMap->getYMax(), theMap->getResolution() *10);

    /* Count the particle number in every cell */
    for(size_t i = 0; i < m_particles.size(); i++)
    {
        int x = x2idx_map.x2idx(m_particles[i].d->x());
        int y = x2idx_map.y2idx(m_particles[i].d->y());
        ++particle_counter[y * x2idx_map.getSizeX() + x];
    }

    /* Judge particle convergency */
    vector< pair<int, int> > particle_counter_vec(particle_counter.begin(), particle_counter.end());
    sort(particle_counter_vec.begin(), particle_counter_vec.end(), big2small);
    int particle_sum = 0;
    if(particle_counter_vec.size() < 5)
    {
        for(size_t i = 0; i < particle_counter_vec.size(); i++)
        {
            particle_sum += particle_counter_vec.at(i).second;
        }
    }
    else
    {
        for(size_t i = 0; i < 5; i++)
        {
            particle_sum += particle_counter_vec.at(i).second;
        }
    }

#if debug
    cout << " top 20 particle numbers: ";
    for(size_t i = 0; i < (particle_counter_vec.size() > 20 ? 20 : particle_counter_vec.size()); i++)
    {
        cout << particle_counter_vec.at(i).second << " ";
    }
    cout << endl;
#endif
    if(particle_counter_vec.size() > 1)
    {
        if(particle_sum > 0 && particle_sum > 0.7 * m_particles.size()
            && particle_counter_vec.at(0).second > particle_counter_vec.at(1).second)
        {
            return true;
        }
        else
        {
            return false;
        }
    }
    else
    {
        return true;
    }
}

size_t  CMonteCarloLocalization2D::calculateFreeSpace(
        COccupancyGridMap2D		        *theMap,
        const double 					down_freeCellsThreshold ,
        const double 					up_freeCellsThreshold ,
        const double 					x_min ,
        const double 					x_max ,
        const double 					y_min ,
        const double 					y_max ,
        const double 					phi_min,
        const double 					phi_max)
{
	MRPT_START

	ASSERT_(theMap!=NULL)

	int					sizeX = theMap->getSizeX();
	int					sizeY = theMap->getSizeY();
	double				gridRes = theMap->getResolution();
	std::vector<double>	freeCells_x,freeCells_y;
	size_t				nFreeCells = 0;
	unsigned int		xIdx1,xIdx2;
	unsigned int		yIdx1,yIdx2;

	if (x_min>theMap->getXMin())
			xIdx1 = max(0, theMap->x2idx( x_min ) );
	else	xIdx1 = 0;
	if (x_max<theMap->getXMax())
			xIdx2 = min(sizeX-1, theMap->x2idx( x_max ) );
	else	xIdx2 = sizeX-1;
	if (y_min>theMap->getYMin())
			yIdx1 = max(0, theMap->y2idx( y_min ) );
	else	yIdx1 = 0;
	if (y_max<theMap->getYMax())
			yIdx2 = min(sizeY-1, theMap->y2idx( y_max ) );
	else	yIdx2 = sizeY-1;

	for (unsigned int x=xIdx1;x<=xIdx2;x++)
		for (unsigned int y=yIdx1;y<=yIdx2;y++)
			if (theMap->getCell(x,y)>=down_freeCellsThreshold
                && theMap->getCell(x,y)<=up_freeCellsThreshold)
			{
                nFreeCells++;
			}

	return nFreeCells;

	MRPT_END
}

bool  CMonteCarloLocalization2D::resetDoorUniformFreeSpace(
	COccupancyGridMap2D		        *theMap,
    const double 					down_freeCellsThreshold ,
	const double 					up_freeCellsThreshold ,
	const double 					x_min ,
	const double 					x_max ,
	const double 					y_min ,
	const double 					y_max ,
	const double 					phi_min,
	const double 					phi_max)
{
	MRPT_START

	ASSERT_(theMap!=NULL)

	int					sizeX = theMap->getSizeX();
	int					sizeY = theMap->getSizeY();
	double				gridRes = theMap->getResolution();
	std::vector<double>	freeCells_x,freeCells_y;
	size_t				nFreeCells;
	unsigned int		xIdx1,xIdx2;
	unsigned int		yIdx1,yIdx2;
	int                 particlesCount;

    freeCells_x.reserve( sizeX * sizeY );
    freeCells_y.reserve( sizeX * sizeY );

	if (x_min>theMap->getXMin())
			xIdx1 = max(0, theMap->x2idx( x_min ) );
	else	xIdx1 = 0;
	if (x_max<theMap->getXMax())
			xIdx2 = min(sizeX-1, theMap->x2idx( x_max ) );
	else	xIdx2 = sizeX-1;
	if (y_min>theMap->getYMin())
			yIdx1 = max(0, theMap->y2idx( y_min ) );
	else	yIdx1 = 0;
	if (y_max<theMap->getYMax())
			yIdx2 = min(sizeY-1, theMap->y2idx( y_max ) );
	else	yIdx2 = sizeY-1;

	for (unsigned int x=xIdx1;x<=xIdx2;x++)
		for (unsigned int y=yIdx1;y<=yIdx2;y++)
			if (theMap->getCell(x,y)>=down_freeCellsThreshold
                && theMap->getCell(x,y)<=up_freeCellsThreshold)
			{
                freeCells_x.push_back(theMap->idx2x(x));
                freeCells_y.push_back(theMap->idx2y(y));
			}

	nFreeCells = freeCells_x.size();
#define DOOR_SAMPLE 1

	// Assure that map is not fully occupied!
	if( nFreeCells )
	{
	    particlesCount = nFreeCells * DOOR_SAMPLE;
        if (particlesCount>0)
        {
            clear();
            m_particles.resize(particlesCount);
            for (int i=0;i<particlesCount;i++)
                m_particles[i].d = new CPose2D();
        }

//        const size_t M = m_particles.size();
        const size_t M = nFreeCells;

        int idx = 0;

        // Generate pose m_particles:
        for (size_t i=0;i<M;i++)
        {
//            int idx = round(randomGenerator.drawUniform(0.0,nFreeCells-1.001));
//            float x = freeCells_x[idx] + randomGenerator.drawUniform( -gridRes, gridRes);
//            float y = freeCells_y[idx] + randomGenerator.drawUniform( -gridRes, gridRes);

            float x = freeCells_x[idx];
            float y = freeCells_y[idx];

            for(size_t j = 0; j < DOOR_SAMPLE; j++)
            {
                m_particles[DOOR_SAMPLE * i + j].d->x(x);
                m_particles[DOOR_SAMPLE * i + j].d->y(y);
                m_particles[DOOR_SAMPLE * i + j].d->phi(M_2PI / DOOR_SAMPLE * j - M_PI);
                m_particles[DOOR_SAMPLE * i + j].log_w = 0;
            }
            idx++;
        }
	    return true;
	}
	else
	{
        return false;
	}

	MRPT_END
}

bool  CMonteCarloLocalization2D::resetFirstUniformFreeSpace(
	COccupancyGridMap2D		        *theMap,
    const double 					down_freeCellsThreshold ,
	const double 					up_freeCellsThreshold ,
	const double 					x_min ,
	const double 					x_max ,
	const double 					y_min ,
	const double 					y_max ,
	const double 					phi_min,
	const double 					phi_max)
{
	MRPT_START

	ASSERT_(theMap!=NULL)

	int					sizeX = theMap->getSizeX();
	int					sizeY = theMap->getSizeY();
	double				gridRes = theMap->getResolution();
	std::vector<double>	freeCells_x,freeCells_y;
	size_t				nFreeCells;
	unsigned int		xIdx1,xIdx2;
	unsigned int		yIdx1,yIdx2;
	int                 particlesCount;

    freeCells_x.reserve( sizeX * sizeY );
    freeCells_y.reserve( sizeX * sizeY );

	if (x_min>theMap->getXMin())
			xIdx1 = max(0, theMap->x2idx( x_min ) );
	else	xIdx1 = 0;
	if (x_max<theMap->getXMax())
			xIdx2 = min(sizeX-1, theMap->x2idx( x_max ) );
	else	xIdx2 = sizeX-1;
	if (y_min>theMap->getYMin())
			yIdx1 = max(0, theMap->y2idx( y_min ) );
	else	yIdx1 = 0;
	if (y_max<theMap->getYMax())
			yIdx2 = min(sizeY-1, theMap->y2idx( y_max ) );
	else	yIdx2 = sizeY-1;

	for (unsigned int x=xIdx1;x<=xIdx2;x++)
		for (unsigned int y=yIdx1;y<=yIdx2;y++)
			if (theMap->getCell(x,y)>=down_freeCellsThreshold
                && theMap->getCell(x,y)<=up_freeCellsThreshold)
			{
                freeCells_x.push_back(theMap->idx2x(x));
                freeCells_y.push_back(theMap->idx2y(y));
			}

	nFreeCells = freeCells_x.size();
#define FIRST_SAMPLE 18

	// Assure that map is not fully occupied!
	if( nFreeCells )
	{
	    particlesCount = nFreeCells * FIRST_SAMPLE;
        if (particlesCount>0)
        {
            clear();
            m_particles.resize(particlesCount);
            for (int i=0;i<particlesCount;i++)
                m_particles[i].d = new CPose2D();
        }

//        const size_t M = m_particles.size();
        const size_t M = nFreeCells;

        int idx = 0;

        // Generate pose m_particles:
        for (size_t i=0;i<M;i++)
        {
//            int idx = round(randomGenerator.drawUniform(0.0,nFreeCells-1.001));
//            float x = freeCells_x[idx] + randomGenerator.drawUniform( -gridRes, gridRes);
//            float y = freeCells_y[idx] + randomGenerator.drawUniform( -gridRes, gridRes);

            float x = freeCells_x[idx];
            float y = freeCells_y[idx];

            for(size_t j = 0; j < FIRST_SAMPLE; j++)
            {
                m_particles[FIRST_SAMPLE * i + j].d->x(x);
                m_particles[FIRST_SAMPLE * i + j].d->y(y);
                m_particles[FIRST_SAMPLE * i + j].d->phi(M_2PI / FIRST_SAMPLE * j - M_PI);
                m_particles[FIRST_SAMPLE * i + j].log_w = 0;
            }
            idx++;
        }
	    return true;
	}
	else
	{
        return false;
	}

	MRPT_END
}

bool  CMonteCarloLocalization2D::resetFirstUniformFreeSpace(
	COccupancyGridMap2D		        *theMap,
    const double 					up_freeCellsThreshold ,
	const double 					down_freeCellsThreshold ,
	const double 					x_min ,
	const double 					x_max ,
	const double 					y_min ,
	const double 					y_max ,
	const double 					phi_min,
	const double 					phi_max,
	const int                       sample_size)
{
	MRPT_START

	ASSERT_(theMap!=NULL)

	int					sizeX = theMap->getSizeX();
	int					sizeY = theMap->getSizeY();
	double				gridRes = theMap->getResolution();
	std::vector<double>	freeCells_x,freeCells_y;
	size_t				nFreeCells;
	unsigned int		xIdx1,xIdx2;
	unsigned int		yIdx1,yIdx2;
	int                 particlesCount;

    freeCells_x.reserve( sizeX * sizeY );
    freeCells_y.reserve( sizeX * sizeY );

	if (x_min>theMap->getXMin())
			xIdx1 = max(0, theMap->x2idx( x_min ) );
	else	xIdx1 = 0;
	if (x_max<theMap->getXMax())
			xIdx2 = min(sizeX-1, theMap->x2idx( x_max ) );
	else	xIdx2 = sizeX-1;
	if (y_min>theMap->getYMin())
			yIdx1 = max(0, theMap->y2idx( y_min ) );
	else	yIdx1 = 0;
	if (y_max<theMap->getYMax())
			yIdx2 = min(sizeY-1, theMap->y2idx( y_max ) );
	else	yIdx2 = sizeY-1;

	for (unsigned int x=xIdx1;x<=xIdx2;x++)
    {
        for (unsigned int y=yIdx1;y<=yIdx2;y++)
        {
            if (theMap->getCell(x,y) > down_freeCellsThreshold)
			{
                freeCells_x.push_back(theMap->idx2x(x));
                freeCells_y.push_back(theMap->idx2y(y));
			}
			else
            {
                unsigned int x_min_sara = x > 3 ? (x - 4) : 0;
                unsigned int x_max_sara = min(sizeX-1, int(x + 4));
                unsigned int y_min_sara = y > 3 ? (y - 4) : 0;
                unsigned int y_max_sara = min(sizeY-1, int(y + 4));

                unsigned int move_count = 0;

                if (theMap->getCell(x, y_min_sara) > down_freeCellsThreshold)move_count++;
                if (theMap->getCell(x, y_max_sara) > down_freeCellsThreshold)move_count++;
                if (theMap->getCell(x_min_sara, y) > down_freeCellsThreshold)move_count++;
                if (theMap->getCell(x_max_sara, y) > down_freeCellsThreshold)move_count++;

                if(move_count > 1)
                {
                    freeCells_x.push_back(theMap->idx2x(x));
                    freeCells_y.push_back(theMap->idx2y(y));
                }
            }
        }
    }

	nFreeCells = freeCells_x.size();

	if( nFreeCells )
	{
	    particlesCount = nFreeCells * sample_size;
        if (particlesCount>0)
        {
            clear();
            m_particles.resize(particlesCount);
        }

        const size_t M = nFreeCells;
        int idx = 0;

        for (size_t i=0;i<M;i++)
        {
            float x = freeCells_x[idx];
            float y = freeCells_y[idx];

            for(size_t j = 0; j < sample_size; j++)
            {
                m_particles[sample_size * i + j].d = new CPose2D();
                m_particles[sample_size * i + j].d->x(x);
                m_particles[sample_size * i + j].d->y(y);
                m_particles[sample_size * i + j].d->phi(M_2PI / sample_size * j - M_PI);
                m_particles[sample_size * i + j].log_w = 0;
            }
            idx++;
        }
	    return true;
	}
	else
	{
        return false;
	}

	MRPT_END
}

bool  CMonteCarloLocalization2D::resetSecondUniformFreeSpace(
	COccupancyGridMap2D		        *theMap,
    const double 					down_freeCellsThreshold ,
	const double 					up_freeCellsThreshold ,
	const double 					x_min ,
	const double 					x_max ,
	const double 					y_min ,
	const double 					y_max ,
	const double 					phi_min,
	const double 					phi_max)
{
	MRPT_START

	ASSERT_(theMap!=NULL)

	int					sizeX = theMap->getSizeX();
	int					sizeY = theMap->getSizeY();
	double				gridRes = theMap->getResolution();
	std::vector<double>	freeCells_x,freeCells_y;
	size_t				nFreeCells;
	unsigned int		xIdx1,xIdx2;
	unsigned int		yIdx1,yIdx2;
	int                 particlesCount;

//    if(sizeX * sizeY > 300)
//    {
//        freeCells_x.reserve( 300 );
//        freeCells_y.reserve( 300 );
//    }
//    else
//    {
        freeCells_x.reserve( sizeX * sizeY );
        freeCells_y.reserve( sizeX * sizeY );
//    }

	if (x_min>theMap->getXMin())
			xIdx1 = max(0, theMap->x2idx( x_min ) );
	else	xIdx1 = 0;
	if (x_max<theMap->getXMax())
			xIdx2 = min(sizeX-1, theMap->x2idx( x_max ) );
	else	xIdx2 = sizeX-1;
	if (y_min>theMap->getYMin())
			yIdx1 = max(0, theMap->y2idx( y_min ) );
	else	yIdx1 = 0;
	if (y_max<theMap->getYMax())
			yIdx2 = min(sizeY-1, theMap->y2idx( y_max ) );
	else	yIdx2 = sizeY-1;

	for (unsigned int x=xIdx1;x<=xIdx2;x++)
		for (unsigned int y=yIdx1;y<=yIdx2;y++)
			if (theMap->getCell(x,y)>=down_freeCellsThreshold
                && theMap->getCell(x,y)<=up_freeCellsThreshold)
			{
//				if(freeCells_x.size() < 300)
//				{
                    freeCells_x.push_back(theMap->idx2x(x));
                    freeCells_y.push_back(theMap->idx2y(y));
//				}
			}

	nFreeCells = freeCells_x.size();
#define SECOND_SAMPLE 18

	// Assure that map is not fully occupied!
	if( nFreeCells )
	{
	    particlesCount = nFreeCells * SECOND_SAMPLE;
        if (particlesCount>0)
        {
            clear();
            m_particles.resize(particlesCount);
            for (int i=0;i<particlesCount;i++)
                m_particles[i].d = new CPose2D();
        }

//        const size_t M = m_particles.size();
        const size_t M = nFreeCells;

        int idx = 0;

        // Generate pose m_particles:
        for (size_t i=0;i<M;i++)
        {
//            int idx = round(randomGenerator.drawUniform(0.0,nFreeCells-1.001));
//            float x = freeCells_x[idx] + randomGenerator.drawUniform( -gridRes, gridRes);
//            float y = freeCells_y[idx] + randomGenerator.drawUniform( -gridRes, gridRes);

            float x = freeCells_x[idx];
            float y = freeCells_y[idx];

            for(size_t j = 0; j < SECOND_SAMPLE; j++)
            {
                m_particles[SECOND_SAMPLE * i + j].d->x(x);
                m_particles[SECOND_SAMPLE * i + j].d->y(y);
                m_particles[SECOND_SAMPLE * i + j].d->phi(M_2PI / SECOND_SAMPLE * j - M_PI);
                m_particles[SECOND_SAMPLE * i + j].log_w = 0;
            }
            idx++;
        }
	    return true;
	}
	else
	{
        return false;
	}

	MRPT_END
}

bool  CMonteCarloLocalization2D::resetPartUniformFreeSpace(
                                COccupancyGridMap2D		        *theMap,
                                const double 					down_freeCellsThreshold ,
                                const double 					up_freeCellsThreshold ,
                                const double 					x_min ,
                                const double 					x_max ,
                                const double 					y_min ,
                                const double 					y_max ,
                                const double 					phi_min,
                                const double 					phi_max)
{
	MRPT_START

	ASSERT_(theMap!=NULL)

	int					sizeX = theMap->getSizeX();
	int					sizeY = theMap->getSizeY();
	double				gridRes = theMap->getResolution();
	std::vector<double>	freeCells_x,freeCells_y;
	size_t				nFreeCells;
	unsigned int		xIdx1,xIdx2;
	unsigned int		yIdx1,yIdx2;
	int                 particlesCount;

    freeCells_x.reserve( sizeX * sizeY );
    freeCells_y.reserve( sizeX * sizeY );

	if (x_min>theMap->getXMin())
			xIdx1 = max(0, theMap->x2idx( x_min ) );
	else	xIdx1 = 0;
	if (x_max<theMap->getXMax())
			xIdx2 = min(sizeX-1, theMap->x2idx( x_max ) );
	else	xIdx2 = sizeX-1;
	if (y_min>theMap->getYMin())
			yIdx1 = max(0, theMap->y2idx( y_min ) );
	else	yIdx1 = 0;
	if (y_max<theMap->getYMax())
			yIdx2 = min(sizeY-1, theMap->y2idx( y_max ) );
	else	yIdx2 = sizeY-1;

	for (unsigned int x=xIdx1;x<=xIdx2;x++)
		for (unsigned int y=yIdx1;y<=yIdx2;y++)
			if (theMap->getCell(x,y)>=down_freeCellsThreshold
                && theMap->getCell(x,y)<=up_freeCellsThreshold)
			{
                freeCells_x.push_back(theMap->idx2x(x));
                freeCells_y.push_back(theMap->idx2y(y));
			}

	nFreeCells = freeCells_x.size();
    #define PART_SAMPLE 24
	// Assure that map is not fully occupied!
	if( nFreeCells )
	{
	    particlesCount = nFreeCells * PART_SAMPLE;
        if (particlesCount>0)
        {
            clear();
            m_particles.resize(particlesCount);
            for (int i=0;i<particlesCount;i++)
                m_particles[i].d = new CPose2D();
        }

        const size_t M = nFreeCells;

        int idx = 0;
        for (size_t i=0;i<M;i++)
        {
            float x = freeCells_x[idx];
            float y = freeCells_y[idx];

            for(size_t j = 0; j < PART_SAMPLE; j++)
            {
                m_particles[PART_SAMPLE * i + j].d->x(x);
                m_particles[PART_SAMPLE * i + j].d->y(y);
                m_particles[PART_SAMPLE * i + j].d->phi(M_2PI / PART_SAMPLE * j - M_PI);
                m_particles[PART_SAMPLE * i + j].log_w = 0;
            }
            idx++;
        }
	    return true;
	}
	else
	{
        return false;
	}

	MRPT_END
}

double CMonteCarloLocalization2D::getMostLikeParticleWeight()
{
	CParticleList::const_iterator	it, itMax=m_particles.begin();
	double		max_w = -1e300;


	for (it=m_particles.begin();it!=m_particles.end();++it)
	{
		if (it->log_w > max_w)
		{
			itMax = it;
			max_w = it->log_w;
		}
	}

	return max_w;
}

void CMonteCarloLocalization2D::setMostLikelyVectorSize(const int  &deque_size)
{
    m_particles_deque.clear();
    for(size_t i = 0; i < deque_size; i++)
    {
        TLikelihoodParam   temp_param;
        temp_param.particle_pose = CPose2D(-100.0, -100.0, 0.0);
        temp_param.particle_logw = -1e300;
        m_particles_deque.push_back(temp_param);
    }

    m_last_particles.particle_pose = CPose2D(-100.0, -100.0, 0.0);
    m_last_particles.particle_logw = -1e300;
}
