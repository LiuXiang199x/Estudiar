/* +---------------------------------------------------------------------------+
   |                     Mobile Robot Programming Toolkit (MRPT)               |
   |                          http://www.mrpt.org/                             |
   |                                                                           |
   | Copyright (c) 2005-2014, Individual contributors, see AUTHORS file        |
   | See: http://www.mrpt.org/Authors - All rights reserved.                   |
   | Released under BSD License. See details in http://www.mrpt.org/License    |
   +---------------------------------------------------------------------------+ */

#include "../slam-precomp.h"   // Precompiled headers

#include <mrpt/utils/CConfigFile.h>
#include <mrpt/poses/CPoint2D.h>
#include <mrpt/poses/CPose2D.h>
#include <mrpt/slam/CMultiMetricMap.h>
#include <mrpt/utils/CStartUpClassesRegister.h>
#include <mrpt/utils/metaprogramming.h>
#include <mrpt/utils/CStream.h>

using namespace mrpt::slam;
using namespace mrpt::utils;
using namespace mrpt::poses;
using namespace mrpt::utils::metaprogramming;

IMPLEMENTS_SERIALIZABLE( CMultiMetricMap, CMetricMap, mrpt::slam )


extern CStartUpClassesRegister  mrpt_slam_class_reg;
const int dumm = mrpt_slam_class_reg.do_nothing(); // Avoid compiler removing this class in static linking

// ------------------------------------------------------------------------
// A few words explaining how all this works:
//  The main hub for operating with all the maps in the internal list
//   if MapExecutor.
//
// All operations go thru MapExecutor::run<OP>() with OP being one of the
//  possible map operations (clear, matching, likelihood, etc.). The
//  idea is that when adding new map types to the internal list of
//  CMultiMetricMap, *only* "MapExecutor" methods must be updated.
// (The only exception are readFromStream() & writeToStream())
//
// The map-specific operations all go into template specializations of
//  other helper structures or in overloaded methods.
//                                                 JLBC (7-AUG-2011)
// ------------------------------------------------------------------------

struct MapExecutor {
	// Apply operation to maps in the same order as declared in CMultiMetricMap.h:
	template <typename OP>
	static void run(const CMultiMetricMap &_mmm, OP &op)
	{
		MRPT_START
		CMultiMetricMap &mmm = const_cast<CMultiMetricMap&>(_mmm);  // This is to avoid duplicating "::run()" for const and non-const.

		for_each( mmm.m_pointsMaps.begin(),mmm.m_pointsMaps.end(), op );
		for_each( mmm.m_gridMaps.begin(),mmm.m_gridMaps.end(),  op );

		MRPT_END
	}

	// Apply operation to maps in the same order as declared in CMultiMetricMap.h:
	template <typename OP>
	static void runOnlyGridMap(const CMultiMetricMap &_mmm, OP &op)
	{
		MRPT_START
		CMultiMetricMap &mmm = const_cast<CMultiMetricMap&>(_mmm);  // This is to avoid duplicating "::run()" for const and non-const.

		//for_each( mmm.m_pointsMaps.begin(),mmm.m_pointsMaps.end(), op );
		for_each( mmm.m_gridMaps.begin(),mmm.m_gridMaps.end(),  op );

		MRPT_END
	}

	// Apply operation to the vectors (or deques) contianing maps:
	template <typename OP>
	static void runOnVectors(const CMultiMetricMap &_mmm, OP &op)
	{
		MRPT_START
		CMultiMetricMap &mmm = const_cast<CMultiMetricMap&>(_mmm);  // This is to avoid duplicating "::run()" for const and non-const.

		op( mmm.m_pointsMaps );
		op( mmm.m_gridMaps );

		MRPT_END
	}
	// Copy all smart pointers:
	static void copyAll(const CMultiMetricMap &other, CMultiMetricMap &mmm) {
		mmm.m_pointsMaps = other.m_pointsMaps;
		mmm.m_gridMaps = other.m_gridMaps;
	}
};  // end of MapExecutor

// ------------------- Begin of map-operations helper templates -------------------
struct MapVectorClearer
{
	template<typename T>
	inline void operator()(T &container) {
		container.clear();
	}
};

// Auxiliary methods are in this base helper struct:
struct MapTraits
{
	const CMultiMetricMap & mmm;
	MapTraits(const CMultiMetricMap & m) : mmm(m) { }

	inline bool isUsedLik(CSimplePointsMapPtr &ptr) {
		return (ptr.present() && (mmm.options.likelihoodMapSelection==CMultiMetricMap::TOptions::mapFuseAll ||
				mmm.options.likelihoodMapSelection==CMultiMetricMap::TOptions::mapPoints ) );
	}
	inline bool isUsedLik(COccupancyGridMap2DPtr &ptr) {
		return (ptr.present() && (mmm.options.likelihoodMapSelection==CMultiMetricMap::TOptions::mapFuseAll ||
				mmm.options.likelihoodMapSelection==CMultiMetricMap::TOptions::mapGrid ) );
	}

	// --------------------
	inline bool isUsedInsert(CSimplePointsMapPtr &ptr) { return ptr.present() && mmm.options.enableInsertion_pointsMap; }
	inline bool isUsedInsert(COccupancyGridMap2DPtr &ptr) { return ptr.present() && mmm.options.enableInsertion_gridMaps; }

}; // end of MapTraits

struct MapComputeLikelihood : public MapTraits
{
	const CObservation    * obs;
	const CPose3D         & takenFrom;
	double                & total_log_lik;

	MapComputeLikelihood(const CMultiMetricMap &m,const CObservation * _obs, const CPose3D & _takenFrom, double & _total_log_lik) :
		MapTraits(m),
		obs(_obs), takenFrom(_takenFrom),
		total_log_lik(_total_log_lik)
	{
		total_log_lik=0;
	}

	template <typename PTR>
	inline void operator()(PTR &ptr) {
		if (isUsedLik(ptr))
			total_log_lik+=ptr->computeObservationLikelihood(obs,takenFrom);
	}

}; // end of MapComputeLikelihood

struct MapCanComputeLikelihood  : public MapTraits
{
	const CObservation    * obs;
	bool                  & can;

	MapCanComputeLikelihood(const CMultiMetricMap &m,const CObservation * _obs, bool & _can) :
		MapTraits(m),
		obs(_obs),
		can(_can)
	{
		can = false;
	}

	template <typename PTR>
	inline void operator()(PTR &ptr) {
		if (isUsedLik(ptr))
			can = can || ptr->canComputeObservationLikelihood(obs);
	}

}; // end of MapCanComputeLikelihood


struct MapInsertObservation : public MapTraits
{
	const CObservation    * obs;
	const CPose3D         * robot_pose;
	int                   & total_insert;

	MapInsertObservation(const CMultiMetricMap &m,const CObservation * _obs, const CPose3D * _robot_pose, int & _total_insert) :
		MapTraits(m),
		obs(_obs), robot_pose(_robot_pose),
		total_insert(_total_insert)
	{
		total_insert = 0;
	}

	template <typename PTR>
	inline void operator()(PTR &ptr) {
		if (isUsedInsert(ptr))
		{
			bool ret = ptr->insertObservation(obs,robot_pose);
			if (ret) total_insert++;
		}
	}
}; // end of MapInsertObservation

struct MapAuxPFCleanup
{
	MapAuxPFCleanup() { }

	template <typename PTR>
	inline void operator()(PTR &ptr) {
		if (ptr.present()) ptr->auxParticleFilterCleanUp();
	}
}; // end of MapAuxPFCleanup


struct MapIsEmpty
{
	bool & is_empty;

	MapIsEmpty(bool & _is_empty) : is_empty(_is_empty)
	{
		is_empty = true;
	}

	template <typename PTR>
	inline void operator()(PTR &ptr) {
		if (ptr.present())
			is_empty = is_empty && ptr->isEmpty();
	}
}; // end of MapIsEmpty

// ------------------- End of map-operations helper templates -------------------



/*---------------------------------------------------------------
			Constructor
  ---------------------------------------------------------------*/
CMultiMetricMap::CMultiMetricMap(
	const TSetOfMetricMapInitializers		*initializers,
	const mrpt::slam::CMultiMetricMap::TOptions	*opts) :
		m_ID(0)
{
	MRPT_START
	MRPT_UNUSED_PARAM(dumm);

	// Create maps
	setListOfMaps(initializers);

	// Do we have initial options?
	if (opts) options = *opts;

	MRPT_END
}

/*---------------------------------------------------------------
			copy constructor
  ---------------------------------------------------------------*/
CMultiMetricMap::CMultiMetricMap(const mrpt::slam::CMultiMetricMap &other ) :
	m_ID(0)
{
	*this = other;	// Call the "=" operator
}

/*---------------------------------------------------------------
			setListOfMaps
  ---------------------------------------------------------------*/
void  CMultiMetricMap::setListOfMaps(
	const mrpt::slam::TSetOfMetricMapInitializers	*initializers )
{
	MRPT_START

	m_ID = 0;

	// Erase current list of maps:
	deleteAllMaps();

	// Do we have any initializer?
	if (initializers!=NULL)
	{
		// The set of options of this class:
		options = initializers->options;


		// Process each entry in the "initializers" and create maps accordingly:
		for (TSetOfMetricMapInitializers::const_iterator it = initializers->begin();it!=initializers->end();++it)
		{
			if ( it->metricMapClassType == CLASS_ID(COccupancyGridMap2D) )
			{
				// -------------------------------------------------------
				//						GRID MAPS
				// -------------------------------------------------------
				COccupancyGridMap2DPtr newGridmap = COccupancyGridMap2DPtr( new COccupancyGridMap2D(
					it->occupancyGridMap2D_options.min_x,
					it->occupancyGridMap2D_options.max_x,
					it->occupancyGridMap2D_options.min_y,
					it->occupancyGridMap2D_options.max_y,
					it->occupancyGridMap2D_options.resolution ) );

				newGridmap->insertionOptions = it->occupancyGridMap2D_options.insertionOpts;
				newGridmap->likelihoodOptions= it->occupancyGridMap2D_options.likelihoodOpts;

				m_gridMaps.push_back( newGridmap );
			}
			else
			if ( it->metricMapClassType == CLASS_ID(CSimplePointsMap) )
			{
				// -------------------------------------------------------
				//						POINTS MAPS
				// -------------------------------------------------------
				CSimplePointsMapPtr newPointsMap = CSimplePointsMap::Create();
				newPointsMap->insertionOptions = it->pointsMapOptions_options.insertionOpts;

				m_pointsMaps.push_back( newPointsMap );
			}
			else
			{
				// -------------------------------------------------------
				//							ERROR
				// -------------------------------------------------------
				THROW_EXCEPTION("Unknown class ID found into initializers (Bug, unsoported map class, or a non-map class?)!!");
			}

		} // end for each "initializers"

	} // end if initializers!=NULL

	MRPT_END
}

/*---------------------------------------------------------------
					clear
  ---------------------------------------------------------------*/
void  CMultiMetricMap::internal_clear()
{
	ObjectClear op;
	MapExecutor::run(*this, op);
}

/*---------------------------------------------------------------
		Copy constructor
  ---------------------------------------------------------------*/
mrpt::slam::CMultiMetricMap & CMultiMetricMap::operator = ( const CMultiMetricMap &other )
{
	MRPT_START

	if (this == &other) return *this;			// Who knows! :-)

	options	  = other.options;
	m_ID	  = other.m_ID;

	// Copy all maps and then make_unique() to really duplicate the objects:
	MapExecutor::copyAll(other,*this);

	// invoke make_unique() operation on each smart pointer:
	ObjectMakeUnique op;
	MapExecutor::run(*this, op);

	return *this;

	MRPT_END
}

/*---------------------------------------------------------------
		Destructor
  ---------------------------------------------------------------*/
CMultiMetricMap::~CMultiMetricMap( )
{
	deleteAllMaps();
}

/*---------------------------------------------------------------
						deleteAllMaps
  ---------------------------------------------------------------*/
void  CMultiMetricMap::deleteAllMaps( )
{
	// invoke make_unique() operation on each smart pointer:
	MapVectorClearer op_vec_clear;
	MapExecutor::runOnVectors(*this, op_vec_clear);

	ObjectMakeUnique op_make_unique;
	MapExecutor::run(*this, op_make_unique);
}

/*---------------------------------------------------------------
  Implements the writing to a CStream capability of CSerializable objects
 ---------------------------------------------------------------*/
void  CMultiMetricMap::writeToStream(CStream &out, int *version) const
{
	if (version)
		*version = 10;
	else
	{
		// Version 5: The options:
		out << options.enableInsertion_pointsMap
			<< options.enableInsertion_gridMaps;
        std::cout << "[CMultiMetricMap] writeToStream options.enableInsertion_pointsMap " << options.enableInsertion_pointsMap << std::endl;
        std::cout << "[CMultiMetricMap] writeToStream options.enableInsertion_gridMaps " << options.enableInsertion_gridMaps << std::endl;

		// The data
		uint32_t	i,n = static_cast<uint32_t>(m_gridMaps.size());


		// grid maps:
		// ----------------------
		out << n;
		for (i=0;i<n;i++)	out << *m_gridMaps[i];
        std::cout << "[CMultiMetricMap] writeToStream options.m_gridMaps " << n << std::endl;

		// Points maps:
		// ----------------------
		n = m_pointsMaps.size();
		out << n;
		for (i=0;i<n;i++)	out << *m_pointsMaps[i];
        std::cout << "[CMultiMetricMap] writeToStream options.m_pointsMaps " << n << std::endl;

		// Added in version 3:
		out << static_cast<uint32_t>(m_ID);
	}
}

/*---------------------------------------------------------------
  Implements the reading from a CStream capability of CSerializable objects
 ---------------------------------------------------------------*/
void  CMultiMetricMap::readFromStream(CStream &in, int version)
{
	switch(version)
	{
	case 0:
	case 1:
	case 2:
	case 3:
	case 4:
	case 5:
	case 6:
	case 7:
	case 8:
	case 9:
	case 10:
		{
		    std::cout << "[CMultiMetricMap] readFromStream version " << version << std::endl;
			uint32_t  n;

			// Version 5: The options:
			if (version>=5)
			{
				in  >> options.enableInsertion_pointsMap
					>> options.enableInsertion_gridMaps;
			}
			else
			{ } // Default!

        std::cout << "[CMultiMetricMap] readFromStream options.enableInsertion_pointsMap " << options.enableInsertion_pointsMap << std::endl;
        std::cout << "[CMultiMetricMap] readFromStream options.enableInsertion_gridMaps " << options.enableInsertion_gridMaps << std::endl;

			// grid maps:
			// ----------------------
			if (version>=2)
			{
				in >> n;
			}
			else
			{
				// Compatibility: Only 1 grid map!
				n = 1;
			}
        std::cout << "[CMultiMetricMap] readFromStream n1 " << n << std::endl;

			// Free previous grid maps:
			m_gridMaps.clear();

			// Load from stream:
			m_gridMaps.resize(n);
			for_each( m_gridMaps.begin(), m_gridMaps.end(), ObjectReadFromStream(&in) );
        std::cout << "[CMultiMetricMap] readFromStream options.m_gridMaps " << m_gridMaps.size() << std::endl;


			// Points maps:
			// ----------------------
			if (version>=2)
						in >> n;
			else		n = 1;			// Compatibility: Always there is a points map!
        std::cout << "[CMultiMetricMap] readFromStream n2 " << n << std::endl;

			m_pointsMaps.clear();
			m_pointsMaps.resize(n);
			for_each(m_pointsMaps.begin(), m_pointsMaps.end(), ObjectReadFromStream(&in) );
        std::cout << "[CMultiMetricMap] readFromStream options.m_pointsMaps " << m_pointsMaps.size() << std::endl;

			if (version>=3)
			{
				uint32_t	ID;
				in >> ID; m_ID = ID;
			}
			else
                m_ID = 0;


		} break;
	default:
		MRPT_THROW_UNKNOWN_SERIALIZATION_VERSION(version)

	};
}


/*---------------------------------------------------------------
 Computes the likelihood that a given observation was taken from a given pose in the world being modeled with this map.
	takenFrom The robot's pose the observation is supposed to be taken from.
	obs The observation.
 This method returns a likelihood in the range [0,1].
 ---------------------------------------------------------------*/
double	 CMultiMetricMap::computeObservationLikelihood(
			const CObservation		*obs,
			const CPose3D			&takenFrom )
{
	double ret_log_lik;
	MapComputeLikelihood op_likelihood(*this,obs,takenFrom,ret_log_lik);

	MapExecutor::runOnlyGridMap(*this,op_likelihood);

	MRPT_CHECK_NORMAL_NUMBER(ret_log_lik)
	return ret_log_lik;
}

/*---------------------------------------------------------------
Returns true if this map is able to compute a sensible likelihood function for this observation (i.e. an occupancy grid map cannot with an image).
\param obs The observation.
 ---------------------------------------------------------------*/
bool CMultiMetricMap::canComputeObservationLikelihood( const CObservation *obs )
{
	bool can_comp;

	MapCanComputeLikelihood op_can_likelihood(*this,obs,can_comp);
	MapExecutor::run(*this,op_can_likelihood);
	return can_comp;
}

/*---------------------------------------------------------------
				getNewStaticPointsRatio
Returns the ratio of points in a map which are new to the points map while falling into yet static cells of gridmap.
	points The set of points to check.
	takenFrom The pose for the reference system of points, in global coordinates of this hybrid map.
 ---------------------------------------------------------------*/
float  CMultiMetricMap::getNewStaticPointsRatio(
		CPointsMap		*points,
		CPose2D			&takenFrom )
{
	const size_t nTotalPoints = points->size();
	ASSERT_( m_gridMaps.size()>0 );

	// There must be points!
	if ( !nTotalPoints ) return 0.0f;

	// Compute matching:
	mrpt::utils::TMatchingPairList correspondences;
	TMatchingExtraResults extraResults;
	TMatchingParams params;
	params.maxDistForCorrespondence = 0.95f*m_gridMaps[0]->insertionOptions.maxDistanceInsertion;

	m_gridMaps[0]->determineMatching2D(
		points,
		takenFrom,
		correspondences,
		params, extraResults);

	size_t nStaticPoints = 0;
	TPoint2D g,l;

	for (size_t i=0;i<nTotalPoints;i++)
	{
		bool	hasCoor = false;
		// Has any correspondence?
		for (mrpt::utils::TMatchingPairList::iterator corrsIt=correspondences.begin();!hasCoor && corrsIt!=correspondences.end();corrsIt++)
			if (corrsIt->other_idx==i)
				hasCoor = true;

		if ( !hasCoor )
		{
			// The distance between the point and the robot: If it is farther than the insertion max. dist.
			//   it should not be consider as an static point!!
			points->getPoint(i,l);

			CPoint2D	temp = CPoint2D(l) - takenFrom;
			if ( temp.norm() < params.maxDistForCorrespondence)
			{
				// A new point
				// ------------------------------------------
				// Translate point to global coordinates:
				g = takenFrom + l;

				if ( m_gridMaps[0]->isStaticPos( g.x, g.y ) )
				{
					// A new, static point:
					nStaticPoints++;
				}
			}
		}
	}	// End of for

	return nStaticPoints/(static_cast<float>(nTotalPoints));
}


/*---------------------------------------------------------------
					insertObservation

Insert the observation information into this map.
 ---------------------------------------------------------------*/
bool  CMultiMetricMap::internal_insertObservation(
		const CObservation	*obs,
		const CPose3D			*robotPose)
{
	int total_insert;
	MapInsertObservation op_insert_obs(*this,obs,robotPose,total_insert);
	MapExecutor::run(*this,op_insert_obs);
	return total_insert!=0;
}

/*---------------------------------------------------------------
					computeMatchingWith2D
 ---------------------------------------------------------------*/
void CMultiMetricMap::determineMatching2D(
	const CMetricMap      * otherMap,
	const CPose2D         & otherMapPose,
	TMatchingPairList     & correspondences,
	const TMatchingParams & params,
	TMatchingExtraResults & extraResults ) const
{
    MRPT_START
	ASSERTMSG_( m_pointsMaps.empty()==1, "There is not exactly 1 points maps in the multimetric map." );
	m_pointsMaps[0]->determineMatching2D(otherMap,otherMapPose,correspondences,params,extraResults);
    MRPT_END
}

/*---------------------------------------------------------------
					isEmpty
 ---------------------------------------------------------------*/
bool  CMultiMetricMap::isEmpty() const
{
	bool is_empty;
	MapIsEmpty op_insert_obs(is_empty);
	MapExecutor::run(*this,op_insert_obs);
	return is_empty;
}

/*---------------------------------------------------------------
					TMetricMapInitializer
 ---------------------------------------------------------------*/
TMetricMapInitializer::TMetricMapInitializer() :
	metricMapClassType(NULL),
	occupancyGridMap2D_options(),
	pointsMapOptions_options()
{
}

/*---------------------------------------------------------------
					TOccGridMap2DOptions
 ---------------------------------------------------------------*/
TMetricMapInitializer::TOccGridMap2DOptions::TOccGridMap2DOptions() :
	min_x(-10.0f),
	max_x(10.0f),
	min_y(-10.0f),
	max_y(10.0f),
	resolution(0.10f),
	insertionOpts(),
	likelihoodOpts()
{
}

/*---------------------------------------------------------------
					CPointsMapOptions
 ---------------------------------------------------------------*/
TMetricMapInitializer::CPointsMapOptions::CPointsMapOptions() :
	insertionOpts()
{

}

/*---------------------------------------------------------------
					CGasConcentrationGridMap2DOptions
 ---------------------------------------------------------------*/
void  CMultiMetricMap::saveMetricMapRepresentationToFile(
	const std::string	&filNamePrefix
	) const
{
	MRPT_START

	unsigned int		idx;

	// grid maps:
	{
		std::deque<COccupancyGridMap2DPtr>::const_iterator	it;
		for (idx=0,it = m_gridMaps.begin();it!=m_gridMaps.end();it++,idx++)
		{
			std::string		fil( filNamePrefix );
			fil += format("_gridmap_no%02u",idx);
			(*it)->saveMetricMapRepresentationToFile( fil );
		}
	}

	// Points maps:
	{
		std::deque<CSimplePointsMapPtr>::const_iterator	it;
		for (idx=0,it = m_pointsMaps.begin();it!=m_pointsMaps.end();it++,idx++)
		{
			std::string		fil( filNamePrefix );
			fil += format("_pointsmap_no%02u",idx);
			(*it)->saveMetricMapRepresentationToFile( fil );
		}
	}

	MRPT_END
}

/*---------------------------------------------------------------
		TSetOfMetricMapInitializers::loadFromConfigFile
 ---------------------------------------------------------------*/
void  TSetOfMetricMapInitializers::loadFromConfigFile(
	const mrpt::utils::CConfigFileBase  &ini,
	const std::string &sectionName )
{
	MRPT_START

	std::string subSectName;

	// Delete previous contents:
	clear();

/*
		  *  ; Creation of maps:
		  *  occupancyGrid_count=<Number of mrpt::slam::COccupancyGridMap2D maps>
		  *  gasGrid_count=<Number of mrpt::slam::CGasConcentrationGridMap2D maps>
		  *  wifiGrid_count=<Number of mrpt::slam::CWirelessPowerGridMap2D maps>
		  *  landmarksMap_count=<0 or 1, for creating a mrpt::slam::CLandmarksMap map>
		  *  beaconMap_count=<0 or 1>
		  *  pointsMap_count=<0 or 1, for creating a mrpt::slam::CSimplePointsMap map>
*/

	unsigned int n = ini.read_int(sectionName,"occupancyGrid_count",0);
	for (unsigned int i=0;i<n;i++)
	{
		TMetricMapInitializer	init;

		init.metricMapClassType					= CLASS_ID( COccupancyGridMap2D );

		// [<sectionName>+"_occupancyGrid_##_creationOpts"]
		subSectName = format("%s_occupancyGrid_%02u_creationOpts",sectionName.c_str(),i);

		init.occupancyGridMap2D_options.min_x	= ini.read_float(subSectName,"min_x",init.occupancyGridMap2D_options.min_x);
		init.occupancyGridMap2D_options.max_x	= ini.read_float(subSectName,"max_x",init.occupancyGridMap2D_options.max_x);
		init.occupancyGridMap2D_options.min_y	= ini.read_float(subSectName,"min_y",init.occupancyGridMap2D_options.min_y);
		init.occupancyGridMap2D_options.max_y	= ini.read_float(subSectName,"max_y",init.occupancyGridMap2D_options.max_y);
		init.occupancyGridMap2D_options.resolution = ini.read_float(subSectName,"resolution",init.occupancyGridMap2D_options.resolution);

		// [<sectionName>+"_occupancyGrid_##_insertOpts"]
		init.occupancyGridMap2D_options.insertionOpts.loadFromConfigFile(ini,format("%s_occupancyGrid_%02u_insertOpts",sectionName.c_str(),i));

		// [<sectionName>+"_occupancyGrid_##_likelihoodOpts"]
		init.occupancyGridMap2D_options.likelihoodOpts.loadFromConfigFile(ini,format("%s_occupancyGrid_%02u_likelihoodOpts",sectionName.c_str(),i));

		// Add the map and its params to the list of "to-create":
		this->push_back(init);
	} // end for i

	n = ini.read_int(sectionName,"pointsMap_count",0);
	for (unsigned int i=0;i<n;i++)
	{
		TMetricMapInitializer	init;

		init.metricMapClassType					= CLASS_ID( CSimplePointsMap );

		// [<sectionName>+"_pointsMap_##_creationOpts"]
		subSectName = format("%s_pointsMap_%02u_creationOpts",sectionName.c_str(),i);

		// [<sectionName>+"_pointsMap_##_insertOpts"]
		init.pointsMapOptions_options.insertionOpts.loadFromConfigFile(ini,format("%s_pointsMap_%02u_insertOpts",sectionName.c_str(),i));

		// [<sectionName>+"_pointsMap_##_likelihoodOpts"]
		init.pointsMapOptions_options.likelihoodOpts.loadFromConfigFile(ini,format("%s_pointsMap_%02u_likelihoodOpts",sectionName.c_str(),i));

		// Add the map and its params to the list of "to-create":
		this->push_back(init);
	} // end for i

/*
		  *  ; Selection of map for likelihood: (occGrid=0, points=1,landmarks=2,gasGrid=3)
		  *  likelihoodMapSelection=<0-3>
*/
	MRPT_LOAD_HERE_CONFIG_VAR_CAST(likelihoodMapSelection,int,CMultiMetricMap::TOptions::TMapSelectionForLikelihood,options.likelihoodMapSelection, ini,sectionName );

/*
		  *  ; Enables (1) / Disables (0) insertion into specific maps:
		  *  enableInsertion_pointsMap=<0/1>
		  *  enableInsertion_landmarksMap=<0/1>
		  *  enableInsertion_gridMaps=<0/1>
		  *  enableInsertion_gasGridMaps=<0/1>
		  *  enableInsertion_wifiGridMaps=<0/1>
		  *  enableInsertion_beaconMap=<0/1>
*/
	MRPT_LOAD_HERE_CONFIG_VAR(enableInsertion_pointsMap,bool,	options.enableInsertion_pointsMap,		ini,sectionName);
	MRPT_LOAD_HERE_CONFIG_VAR(enableInsertion_gridMaps,bool,		options.enableInsertion_gridMaps,		ini,sectionName);

	MRPT_END
}

/*---------------------------------------------------------------
		TSetOfMetricMapInitializers::dumpToTextStream
 ---------------------------------------------------------------*/
void  TSetOfMetricMapInitializers::dumpToTextStream(CStream	&out) const
{
	MRPT_START

	out.printf("====================================================================\n\n");
	out.printf("             Set of internal maps for 'CMultiMetricMap' object\n\n");
	out.printf("====================================================================\n");

	out.printf("likelihoodMapSelection                  = %s\n",
		TEnumType<CMultiMetricMap::TOptions::TMapSelectionForLikelihood>::value2name(options.likelihoodMapSelection).c_str() );

	LOADABLEOPTS_DUMP_VAR(options.enableInsertion_pointsMap		, bool)
	LOADABLEOPTS_DUMP_VAR(options.enableInsertion_gridMaps		, bool)

	// Show each map:
	out.printf("Showing next the %u internal maps:\n\n", (int)size());

	int i=0;
	for (const_iterator it=begin();it!=end();++it,i++)
	{
		out.printf("------------------------- Internal map %u out of %u --------------------------\n",i+1,(int)size());

		out.printf("                 C++ Class: '%s'\n", it->metricMapClassType->className);

		if (it->metricMapClassType==CLASS_ID(COccupancyGridMap2D))
		{
			out.printf("resolution                              = %0.3f\n",it->occupancyGridMap2D_options.resolution);
			out.printf("min_x                                   = %0.3f\n",it->occupancyGridMap2D_options.min_x);
			out.printf("max_x                                   = %0.3f\n",it->occupancyGridMap2D_options.max_x);
			out.printf("min_y                                   = %0.3f\n",it->occupancyGridMap2D_options.min_y);
			out.printf("max_y                                   = %0.3f\n",it->occupancyGridMap2D_options.max_y);

			it->occupancyGridMap2D_options.insertionOpts.dumpToTextStream(out);
			it->occupancyGridMap2D_options.likelihoodOpts.dumpToTextStream(out);
		}
		else
		if (it->metricMapClassType==CLASS_ID(CSimplePointsMap))
		{
			it->pointsMapOptions_options.insertionOpts.dumpToTextStream(out);
			it->pointsMapOptions_options.likelihoodOpts.dumpToTextStream(out);
		}
		else
		{
			THROW_EXCEPTION_CUSTOM_MSG1("Unknown class!: '%s'",it->metricMapClassType->className);
		}
	} // for "it"

	MRPT_END
}

/*---------------------------------------------------------------
 Computes the ratio in [0,1] of correspondences between "this" and the "otherMap" map, whose 6D pose relative to "this" is "otherMapPose"
 *   In the case of a multi-metric map, this returns the average between the maps. This method always return 0 for grid maps.
 * \param  otherMap					  [IN] The other map to compute the matching with.
 * \param  otherMapPose				  [IN] The 6D pose of the other map as seen from "this".
 * \param  maxDistForCorr			  [IN] The minimum distance between 2 non-probabilistic map elements for counting them as a correspondence.
 * \param  maxMahaDistForCorr		  [IN] The minimum Mahalanobis distance between 2 probabilistic map elements for counting them as a correspondence.
 *
 * \return The matching ratio [0,1]
 * \sa computeMatchingWith2D
  ---------------------------------------------------------------*/
float  CMultiMetricMap::compute3DMatchingRatio(
		const CMetricMap								*otherMap,
		const CPose3D							&otherMapPose,
		float									maxDistForCorr,
		float									maxMahaDistForCorr
		) const
{
	MRPT_START

	size_t		nMapsComputed = 0;
	float		accumResult = 0;

	// grid maps: NO

	// Gas grids maps: NO

	// Wifi grids maps: NO

	// Points maps:
	if (m_pointsMaps.size()>0)
	{
		ASSERT_(m_pointsMaps.size()==1);
		accumResult += m_pointsMaps[0]->compute3DMatchingRatio( otherMap, otherMapPose,maxDistForCorr,maxMahaDistForCorr );
		nMapsComputed++;
	}

	// Return average:
	if (nMapsComputed) accumResult/=nMapsComputed;
	return accumResult;

	MRPT_END
}

/*---------------------------------------------------------------
					auxParticleFilterCleanUp
 ---------------------------------------------------------------*/
void  CMultiMetricMap::auxParticleFilterCleanUp()
{
	MRPT_START
	MapAuxPFCleanup op_cleanup;
	MapExecutor::run(*this,op_cleanup);
	MRPT_END
}



/** Load parameters from configuration source
  */
void  CMultiMetricMap::TOptions::loadFromConfigFile(
	const mrpt::utils::CConfigFileBase	&source,
	const std::string		&section)
{
	likelihoodMapSelection = source.read_enum<TMapSelectionForLikelihood>(section,"likelihoodMapSelection",likelihoodMapSelection);

	MRPT_LOAD_CONFIG_VAR(enableInsertion_pointsMap, bool,  source, section );
	MRPT_LOAD_CONFIG_VAR(enableInsertion_gridMaps, bool,  source, section );


}

/** This method must display clearly all the contents of the structure in textual form, sending it to a CStream.
  */
void  CMultiMetricMap::TOptions::dumpToTextStream(CStream	&out) const
{
	out.printf("\n----------- [CMultiMetricMap::TOptions] ------------ \n\n");

	out.printf("likelihoodMapSelection                  = %i\n",	static_cast<int>(likelihoodMapSelection) );
	out.printf("enableInsertion_pointsMap               = %c\n",	enableInsertion_pointsMap ? 'Y':'N');
	out.printf("enableInsertion_gridMaps                = %c\n",	enableInsertion_gridMaps ? 'Y':'N');

	out.printf("\n");
}


/** If the map is a simple points map or it's a multi-metric map that contains EXACTLY one simple points map, return it.
* Otherwise, return NULL
*/
const CSimplePointsMap * CMultiMetricMap::getAsSimplePointsMap() const
{
	MRPT_START
	ASSERT_(m_pointsMaps.size()==1 || m_pointsMaps.size()==0)
	if (m_pointsMaps.empty()) return NULL;
	else return m_pointsMaps[0].pointer();
	MRPT_END
}
CSimplePointsMap * CMultiMetricMap::getAsSimplePointsMap()
{
	MRPT_START
	ASSERT_(m_pointsMaps.size()==1 || m_pointsMaps.size()==0)
	if (m_pointsMaps.empty()) return NULL;
	else return m_pointsMaps[0].pointer();
	MRPT_END
}



/** Ctor: TOptions::TOptions
*/
CMultiMetricMap::TOptions::TOptions() :
	likelihoodMapSelection(mapFuseAll),
	enableInsertion_pointsMap  (true),
	enableInsertion_gridMaps(true)
{
}
