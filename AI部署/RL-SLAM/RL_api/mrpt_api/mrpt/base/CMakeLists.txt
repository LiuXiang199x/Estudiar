# --------------------------------------------------------------------------
#   The list of "libs" which can be included can be found in:
#     http://www.mrpt.org/Libraries
# --------------------------------------------------------------------------
PROJECT(mrpt_base)

# SOURCE FILES
ADD_DIR_SRCS(MRPT_BASE_SOURCES ${MRPT_LIBS_ROOT}/base/src/bayes ".cpp")
ADD_DIR_SRCS(MRPT_BASE_SOURCES ${MRPT_LIBS_ROOT}/base/include/mrpt/bayes ".h")

ADD_DIR_SRCS(MRPT_BASE_SOURCES ${MRPT_LIBS_ROOT}/base/src/compress ".cpp")
ADD_DIR_SRCS(MRPT_BASE_SOURCES ${MRPT_LIBS_ROOT}/base/include/mrpt/compress ".h")

ADD_DIR_SRCS(MRPT_BASE_SOURCES ${MRPT_LIBS_ROOT}/base/src/math ".cpp")
ADD_DIR_SRCS(MRPT_BASE_SOURCES ${MRPT_LIBS_ROOT}/base/include/mrpt/math ".h")

ADD_DIR_SRCS(MRPT_BASE_SOURCES ${MRPT_LIBS_ROOT}/base/src/poses ".cpp")
ADD_DIR_SRCS(MRPT_BASE_SOURCES ${MRPT_LIBS_ROOT}/base/include/mrpt/poses ".h")

ADD_DIR_SRCS(MRPT_BASE_SOURCES ${MRPT_LIBS_ROOT}/base/src/random ".cpp")
ADD_DIR_SRCS(MRPT_BASE_SOURCES ${MRPT_LIBS_ROOT}/base/include/mrpt/random ".h")

ADD_DIR_SRCS(MRPT_BASE_SOURCES ${MRPT_LIBS_ROOT}/base/src/synch ".cpp")
ADD_DIR_SRCS(MRPT_BASE_SOURCES ${MRPT_LIBS_ROOT}/base/include/mrpt/synch ".h")

ADD_DIR_SRCS(MRPT_BASE_SOURCES ${MRPT_LIBS_ROOT}/base/src/system ".cpp")
ADD_DIR_SRCS(MRPT_BASE_SOURCES ${MRPT_LIBS_ROOT}/base/include/mrpt/system ".h")

ADD_DIR_SRCS(MRPT_BASE_SOURCES ${MRPT_LIBS_ROOT}/base/src/utils ".cpp")
ADD_DIR_SRCS(MRPT_BASE_SOURCES ${MRPT_LIBS_ROOT}/base/include/mrpt/utils ".h")

ADD_DIR_SRCS(MRPT_BASE_SOURCES ${MRPT_LIBS_ROOT}/base/src ".cpp")
ADD_DIR_SRCS(MRPT_BASE_SOURCES ${MRPT_LIBS_ROOT}/base/include/mrpt ".h")

ADD_DIR_SRCS(MRPT_BASE_SOURCES ${MRPT_LIBS_ROOT}/base/src/utils/simpleini ".c")
ADD_DIR_SRCS(MRPT_BASE_SOURCES ${MRPT_LIBS_ROOT}/base/src/mrpt/utils/simpleini ".h")

ADD_DIR_SRCS(MRPT_BASE_SOURCES ${MRPT_LIBS_ROOT}/base/src/utils/jpeglib ".cpp")
ADD_DIR_SRCS(MRPT_BASE_SOURCES ${MRPT_LIBS_ROOT}/base/src/mrpt/utils/jpeglib ".h")

ADD_DIR_SRCS(MRPT_BASE_SOURCES ${MRPT_LIBS_ROOT}/base/src/utils/xmlparser ".cpp")
ADD_DIR_SRCS(MRPT_BASE_SOURCES ${MRPT_LIBS_ROOT}/base/src/mrpt/utils/xmlparser ".h")

ADD_DIR_SRCS(MRPT_BASE_SOURCES ${MRPT_LIBS_ROOT}/base/src/math/CSparse ".c")
ADD_DIR_SRCS(MRPT_BASE_SOURCES ${MRPT_LIBS_ROOT}/base/src/mrpt/math/CSparse ".h")

ADD_DIR_SRCS(MRPT_BASE_SOURCES ${MRPT_LIBS_ROOT}/base/src/math/kmeans++ ".cpp")
ADD_DIR_SRCS(MRPT_BASE_SOURCES ${MRPT_LIBS_ROOT}/base/src/mrpt/math/kmeans++ ".h")

ADD_DIR_SRCS(MRPT_BASE_SOURCES ${MRPT_LIBS_ROOT}/base/src/compress/zlib ".c")
ADD_DIR_SRCS(MRPT_BASE_SOURCES ${MRPT_LIBS_ROOT}/base/src/mrpt/compress/zlib  ".h")

#FILE(GLOB EIGEN_SOURCE_LIST ${MRPT_LIBS_ROOT}/base/include/eigen/*)
#LIST(APPEND MRPT_BASE_SOURCES ${EIGEN_SOURCE_LIST})

# REMOVE UNIT TEST FILE
REMOVE_MATCHING_FILES_FROM_LIST(".*_unittest.cpp" MRPT_BASE_SOURCES)

# BUILD LIABARY WITH ALL SOURCES FILE 
ADD_LIBRARY(mrpt-base SHARED ${MRPT_BASE_SOURCES})
TARGET_LINK_LIBRARIES(mrpt-base ${GNUSTL_SHARED_LIB} ${RT_PTHREAD_LIB})

# SELECT SHARE VERSION
if(VERSION)
    message("SET LIBS VERSION ${MRPT_VERSION}.${MRPT_SOVERSION}")
    SET_TARGET_PROPERTIES(mrpt-base PROPERTIES VERSION ${MRPT_VERSION} SOVERSION ${MRPT_SOVERSION})
endif()

# SET LIBRARY OUTPUT PATH
SET(LIBRARY_OUTPUT_PATH ${LIB_OUT_DIR})

