# --------------------------------------------------------------------------
#   The list of "libs" which can be included can be found in:
#     http://www.mrpt.org/Libraries
# --------------------------------------------------------------------------
PROJECT(mrpt_obs)

# SOURCE FILES
ADD_DIR_SRCS(OBS_SOURCE_LIST ${MRPT_LIBS_ROOT}/obs/src ".cpp")
ADD_DIR_SRCS(OBS_SOURCE_LIST ${MRPT_LIBS_ROOT}/obs/include ".h")

# REMOVE UNIT TEST FILE
REMOVE_MATCHING_FILES_FROM_LIST(".*_unittest.cpp" OBS_SOURCE_LIST)

# BUILD LIABARY WITH ALL SOURCES FILE 
ADD_LIBRARY(mrpt-obs SHARED ${OBS_SOURCE_LIST})
ADD_DEPENDENCIES(mrpt-obs mrpt-base)
TARGET_LINK_LIBRARIES(mrpt-obs mrpt-base)

# SELECT SHARE VERSION
if(VERSION)
    SET_TARGET_PROPERTIES(mrpt-obs PROPERTIES VERSION ${mrpt_obs_main_ver} SOVERSION ${mrpt_obs_slave_ver})
endif()

# SET LIBRARY OUTPUT PATH
SET(LIBRARY_OUTPUT_PATH ${LIB_OUT_DIR})
