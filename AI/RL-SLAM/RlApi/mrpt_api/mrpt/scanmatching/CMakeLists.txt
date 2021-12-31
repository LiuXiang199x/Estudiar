# --------------------------------------------------------------------------
#   The list of "libs" which can be included can be found in:
#     http://www.mrpt.org/Libraries
# --------------------------------------------------------------------------
PROJECT(mrpt_scanmatching)

# SOURCE FILES
ADD_DIR_SRCS(SCANMATCHING_SOURCE_LIST ${MRPT_LIBS_ROOT}/scanmatching/src ".cpp")
ADD_DIR_SRCS(SCANMATCHING_SOURCE_LIST ${MRPT_LIBS_ROOT}/scanmatching/include ".h")

# REMOVE UNIT TEST FILE
REMOVE_MATCHING_FILES_FROM_LIST(".*_unittest.cpp" SCANMATCHING_SOURCE_LIST)

# BUILD LIABARY WITH ALL SOURCES FILE 
ADD_LIBRARY(mrpt-scanmatching SHARED ${SCANMATCHING_SOURCE_LIST})
ADD_DEPENDENCIES(mrpt-scanmatching mrpt-base)
TARGET_LINK_LIBRARIES(mrpt-scanmatching mrpt-base)

# SELECT SHARE VERSION
if(VERSION)
    SET_TARGET_PROPERTIES(mrpt-scanmatching PROPERTIES VERSION ${MRPT_VERSION} SOVERSION ${MRPT_SOVERSION})
endif()

# SET LIBRARY OUTPUT PATH
SET(LIBRARY_OUTPUT_PATH ${LIB_OUT_DIR})
