# --------------------------------------------------------------------------
#   The list of "libs" which can be included can be found in:
#     http://www.mrpt.org/Libraries
# --------------------------------------------------------------------------
PROJECT(mrpt_slam)

# BUILD LIABARY WITH ALL SOURCES FILE 
add_dir_srcs(SLAM_SOURCE_LIST ${MRPT_LIBS_ROOT}/slam/src ".cpp")
add_dir_srcs(SLAM_SOURCE_LIST ${MRPT_LIBS_ROOT}/slam/include ".h")
add_dir_srcs(SLAM_SOURCE_LIST ${MRPT_LIBS_ROOT}/slam/src/slam ".cpp")
add_dir_srcs(SLAM_SOURCE_LIST ${MRPT_LIBS_ROOT}/slam/src/maps ".cpp")

# REMOVE UNIT TEST FILE
REMOVE_MATCHING_FILES_FROM_LIST(".*_unittest.cpp" SLAM_SOURCE_LIST)

# BUILD LIABARY WITH ALL SOURCES FILE 
ADD_LIBRARY(mrpt-slam SHARED ${SLAM_SOURCE_LIST})
ADD_DEPENDENCIES(mrpt-slam mrpt-hwdrivers mrpt-scanmatching)
TARGET_LINK_LIBRARIES(mrpt-slam mrpt-hwdrivers mrpt-scanmatching)

# SELECT SHARE VERSION
if(VERSION)
    SET_TARGET_PROPERTIES(mrpt-slam PROPERTIES VERSION ${MRPT_VERSION} SOVERSION ${MRPT_SOVERSION})
endif()

# SET LIBRARY OUTPUT PATH
SET(LIBRARY_OUTPUT_PATH ${LIB_OUT_DIR})
