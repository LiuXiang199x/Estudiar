/* +---------------------------------------------------------------------------+
   |                     Mobile Robot Programming Toolkit (MRPT)               |
   |                          http://www.mrpt.org/                             |
   |                                                                           |
   | Copyright (c) 2005-2014, Individual contributors, see AUTHORS file        |
   | See: http://www.mrpt.org/Authors - All rights reserved.                   |
   | Released under BSD License. See details in http://www.mrpt.org/License    |
   +---------------------------------------------------------------------------+ */

#pragma once

#include <vector>  // For <mrpt-eigen/Eigen/StdVector>
#include <deque>   // For <mrpt-eigen/Eigen/StdDeque>

// needed here for a few basic types used in Eigen MRPT's plugin:
#include <mrpt/math/math_frwds.h>

// --------------------------------------------------
// Include the Eigen3 library headers, including
//  MRPT's extensions:
// --------------------------------------------------
#include <fstream> // These headers are assumed by <mrpt/math/eigen_plugins.h>:
#include <ctime>
#include <stdexcept>
#ifdef EIGEN_MATRIXBASE_H
#	error **FATAL ERROR**: MRPT headers must be included before <mrpt-eigen/Eigen/Dense> headers.
#endif
#ifndef EIGEN_USE_NEW_STDVECTOR
#  define EIGEN_USE_NEW_STDVECTOR
#endif
#include <mrpt-eigen/Eigen/Dense>
#include <mrpt-eigen/Eigen/StdVector>
#include <mrpt-eigen/Eigen/StdDeque>

#if !EIGEN_MRPT_VERSION_AT_LEAST(2,90,0)
#error MRPT needs version 3.0.0-beta of Eigen or newer
#endif

// Template implementations that need to be after all Eigen includes:
#include EIGEN_MATRIXBASE_PLUGIN_POST_IMPL
// --------------------------------------------------
//  End of Eigen includes
// --------------------------------------------------


// This must be put inside any MRPT class that inherits from an Eigen class:
#define MRPT_EIGEN_DERIVED_CLASS_CTOR_OPERATOR_EQUAL(_CLASS_) \
	/*! Assignment operator from any other Eigen class */ \
    template<typename OtherDerived> \
    inline mrpt_autotype & operator= (const mrpt_Eigen::MatrixBase <OtherDerived>& other) { \
        Base::operator=(other); \
        return *this; \
    } \
	/*! Constructor from any other Eigen class */ \
    template<typename OtherDerived> \
	inline _CLASS_(const mrpt_Eigen::MatrixBase <OtherDerived>& other) : Base(other.template cast<typename Base::Scalar>()) { } \

namespace mrpt
{
	namespace math
	{
		/** Column vector, like mrpt_Eigen::MatrixX*, but automatically initialized to zeros since construction */
		template <typename T>
		class dynamic_vector : public mrpt_Eigen::Matrix<T,mrpt_Eigen::Dynamic,1>
		{
		public:
			typedef mrpt_Eigen::Matrix<T,mrpt_Eigen::Dynamic,1> Base;
			typedef dynamic_vector<T> mrpt_autotype;
			typedef T value_type;
			MRPT_MATRIX_CONSTRUCTORS_FROM_POSES(dynamic_vector)
			MRPT_EIGEN_DERIVED_CLASS_CTOR_OPERATOR_EQUAL(dynamic_vector)

			/** Default constructor (vector of given size set to zero) */
			inline dynamic_vector(size_t length=0) { Base::setZero(length); }
			/** Constructor to given size and all entries to some value */
			inline dynamic_vector(size_t length, float value) { Base::resize(length); Base::setConstant(value); }

		};

		typedef dynamic_vector<float>  CVectorFloat;  //!< Column vector, like mrpt_Eigen::MatrixXf, but automatically initialized to zeros since construction
		typedef dynamic_vector<double> CVectorDouble; //!< Column vector, like mrpt_Eigen::MatrixXd, but automatically initialized to zeros since construction
	}

	namespace utils
	{
		class CStream;

		CStream BASE_IMPEXP & operator<<(CStream&s, const mrpt::math::CVectorFloat  &a);
		CStream BASE_IMPEXP & operator<<(CStream&s, const mrpt::math::CVectorDouble &a);
		CStream BASE_IMPEXP & operator>>(CStream&in, mrpt::math::CVectorDouble &a);
		CStream BASE_IMPEXP & operator>>(CStream&in, mrpt::math::CVectorFloat &a);
	}
}
