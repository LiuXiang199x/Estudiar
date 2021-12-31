# --------------------------------------------------------------------------
#   The list of "libs" which can be included can be found in:
#     http://www.mrpt.org/Libraries
# --------------------------------------------------------------------------
PROJECT(mrpt_maps)

# SOURCE FILES
ADD_DIR_SRCS(MAPS_SOURCE_LIST ${MRPT_LIBS_ROOT}/maps/src ".cpp")
ADD_DIR_SRCS(MAPS_SOURCE_LIST ${MRPT_LIBS_ROOT}/maps/include ".h")
ADD_DIR_SRCS(MAPS_SOURCE_LIST ${MRPT_LIBS_ROOT}/slam/include ".h")
ADD_DIR_SRCS(MAPS_SOURCE_LIST ${MRPT_LIBS_ROOT}/maps/src/maps ".cpp")

# REMOVE UNIT TEST FILE
REMOVE_MATCHING_FILES_FROM_LIST(".*_unittest.cpp" MAPS_SOURCE_LIST)

# BUILD LIABARY WITH ALL SOURCES FILE 
ADD_LIBRARY(mrpt-maps SHARED ${MAPS_SOURCE_LIST})
ADD_DEPENDENCIES(mrpt-maps mrpt-obs)
TARGET_LINK_LIBRARIES(mrpt-maps mrpt-obs)

# SELECT SHARE VERSION
if(VERSION)
    SET_TARGET_PROPERTIES(mrpt-maps PROPERTIES VERSION ${MRPT_VERSION} SOVERSION ${MRPT_SOVERSION})
endif()

# SET LIBRARY OUTPUT PATH
SET(LIBRARY_OUTPUT_PATH ${LIB_OUT_DIR})
