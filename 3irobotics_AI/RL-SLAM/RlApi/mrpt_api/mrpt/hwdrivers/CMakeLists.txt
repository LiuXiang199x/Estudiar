# --------------------------------------------------------------------------
#   The list of "libs" which can be included can be found in:
#     http://www.mrpt.org/Libraries
# --------------------------------------------------------------------------
PROJECT(mrpt_hwdrivers)

# SOURCE FILES
ADD_DIR_SRCS(HWDRIVERS_SOURCE_LIST ${MRPT_LIBS_ROOT}/hwdrivers/src ".cpp")
ADD_DIR_SRCS(HWDRIVERS_SOURCE_LIST ${MRPT_LIBS_ROOT}/hwdrivers/include ".h")

ADD_DIR_SRCS(HWDRIVERS_SOURCE_LIST ${MRPT_LIBS_ROOT}/hwdrivers/src/rplidar/src ".cpp")
ADD_DIR_SRCS(HWDRIVERS_SOURCE_LIST ${MRPT_LIBS_ROOT}/hwdrivers/src/rplidar/include ".h")

ADD_DIR_SRCS(HWDRIVERS_SOURCE_LIST ${MRPT_LIBS_ROOT}/hwdrivers/src/rplidar/src/hal ".cpp")
ADD_DIR_SRCS(HWDRIVERS_SOURCE_LIST ${MRPT_LIBS_ROOT}/hwdrivers/src/rplidar/src/hal ".h")

ADD_DIR_SRCS(HWDRIVERS_SOURCE_LIST ${MRPT_LIBS_ROOT}/hwdrivers/src/rplidar/src/arch/linux ".cpp")
ADD_DIR_SRCS(HWDRIVERS_SOURCE_LIST ${MRPT_LIBS_ROOT}/hwdrivers/src/rplidar/src/arch/linux ".h")

# REMOVE UNIT TEST FILE
REMOVE_MATCHING_FILES_FROM_LIST(".*_unittest.cpp" HWDRIVERS_SOURCE_LIST)

# BUILD LIABARY WITH ALL SOURCES FILE 
ADD_LIBRARY(mrpt-hwdrivers SHARED ${HWDRIVERS_SOURCE_LIST})
ADD_DEPENDENCIES(mrpt-hwdrivers mrpt-maps)
TARGET_LINK_LIBRARIES(mrpt-hwdrivers mrpt-maps ${RT_PTHREAD_LIB})

# SELECT SHARE VERSION
if(VERSION)
    SET_TARGET_PROPERTIES(mrpt-hwdrivers PROPERTIES VERSION ${MRPT_VERSION} SOVERSION ${MRPT_SOVERSION})
endif()

# SET LIBRARY OUTPUT PATH
SET(LIBRARY_OUTPUT_PATH ${LIB_OUT_DIR})
