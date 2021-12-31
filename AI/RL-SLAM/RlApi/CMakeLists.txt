cmake_minimum_required(VERSION 3.4.1)

project(RlApi)

set(CMAKE_SYSTEM_NAME Linux)
add_definitions(-w)
# for aarch64
#set(CROSS_COMPILE_TOOL ${CMAKE_SOURCE_DIR}/../../../../../../prebuilts/gcc/linux-x86/aarch64/gcc-linaro-6.3.1-2017.05-x86_64_aarch64-linux-gnu/bin/aarch64-linux-gnu)

# for armhf
# set(CROSS_COMPILE_TOOL ${CMAKE_SOURCE_DIR}/../../../../../../prebuilts/gcc/linux-x86/arm/gcc-linaro-6.3.1-2017.05-x86_64_arm-linux-gnueabihf/bin/arm-linux-gnueabihf)

#1126
#set(CROSS_COMPILE_TOOL /home/zkj/Documents/Windows_Share/Windows_share/1126-cross-compile-tool/gcc-arm-linux-gnueabihf/bin/arm-linux-gnueabihf)

set(CROSS_COMPILE_TOOL /home/xiang/gcc-linaro-6.3.1-2017.05-x86_64_arm-linux-gnueabihf-master/bin/arm-linux-gnueabihf)

# set(CMAKE_C_COMPILER ${CROSS_COMPILE_TOOL}-gcc)
# set(CMAKE_CXX_COMPILER ${CROSS_COMPILE_TOOL}-g++)

#1806/1808
set(CMAKE_C_COMPILER aarch64-linux-gnu-gcc)
set(CMAKE_CXX_COMPILER aarch64-linux-gnu-g++)

#SET(CMAKE_C_COMPILER "arm-linux-gnueabihf-gcc")
#SET(CMAKE_CXX_COMPILER "arm-linux-gnueabihf-g++")


set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

if (CMAKE_C_COMPILER MATCHES "aarch64")
  MESSAGE(STATUS "64--------------")
  set(LIB_ARCH lib64)
else()
  MESSAGE(STATUS "32--------------")
  set(LIB_ARCH lib)
endif()

# rknn so
set(RKNN_API_PATH /home/xiang/Estudiar/AI/RL-SLAM/RlApi/librknn_api)
# include_directories(${RKNN_API_PATH}/include)
set(RKNN_API_LIB ${RKNN_API_PATH}/${LIB_ARCH}/librknn_api.so)

message(${RKNN_API_LIB})
# Rl api & rknn api
set(RL_API_PATH /home/xiang/Estudiar/AI/RL-SLAM)
include_directories(${RL_API_PATH})

include_directories(/home/xiang/Estudiar/AI/RL-SLAM/RlApi)
include_directories(/home/xiang/Estudiar/AI/RL-SLAM/RlApi/mrpt_api/mrpt/maps/include)

# mrpt api
############## mrpt_base #################
set(MRPT_PATH /home/xiang/Estudiar/AI/RL-SLAM/RlApi/mrpt_api)
include_directories(${MRPT_PATH}/mrpt/base/include/)
include_directories(${MRPT_PATH}/mrpt/base/include/gtest/)
include_directories(${MRPT_PATH}/mrpt/base/include/gtest/internal/)

include_directories(${MRPT_PATH}/mrpt/base/include/mrpt/)
include_directories(${MRPT_PATH}/mrpt/base/include/mrpt/base/)
include_directories(${MRPT_PATH}/mrpt/base/include/mrpt/bayes/)
include_directories(${MRPT_PATH}/mrpt/base/include/mrpt/compress/)
include_directories(${MRPT_PATH}/mrpt/base/include/mrpt/math/)
include_directories(${MRPT_PATH}/mrpt/base/include/mrpt/otherlibs/)
include_directories(${MRPT_PATH}/mrpt/base/include/mrpt/otherlibs/CSparse/)
include_directories(${MRPT_PATH}/mrpt/base/include/mrpt/otherlibs/nanoflann/)
include_directories(${MRPT_PATH}/mrpt/base/include/mrpt/otherlibs/stlplus/)
include_directories(${MRPT_PATH}/mrpt/base/include/mrpt/otherlibs/tclap/)
include_directories(${MRPT_PATH}/mrpt/base/include/mrpt/poses/)
include_directories(${MRPT_PATH}/mrpt/base/include/mrpt/random/)
include_directories(${MRPT_PATH}/mrpt/base/include/mrpt/synch/)
include_directories(${MRPT_PATH}/mrpt/base/include/mrpt/system/)
include_directories(${MRPT_PATH}/mrpt/base/include/mrpt/utils/)

include_directories(${MRPT_PATH}/mrpt/base/include/mrpt-eigen/)
include_directories(${MRPT_PATH}/mrpt/base/include/mrpt-eigen/Eigen/)
include_directories(${MRPT_PATH}/mrpt/base/include/mrpt-eigen/Eigen/src/)
include_directories(${MRPT_PATH}/mrpt/base/include/mrpt-eigen/Eigen/src/Cholesky/)
include_directories(${MRPT_PATH}/mrpt/base/include/mrpt-eigen/Eigen/src/CholmodSupport/)
include_directories(${MRPT_PATH}/mrpt/base/include/mrpt-eigen/Eigen/src/Core/)
include_directories(${MRPT_PATH}/mrpt/base/include/mrpt-eigen/Eigen/src/Core/arch/)
include_directories(${MRPT_PATH}/mrpt/base/include/mrpt-eigen/Eigen/src/Core/arch/AltiVec/)
include_directories(${MRPT_PATH}/mrpt/base/include/mrpt-eigen/Eigen/src/Core/arch/Default/)
include_directories(${MRPT_PATH}/mrpt/base/include/mrpt-eigen/Eigen/src/Core/arch/NEON/)
include_directories(${MRPT_PATH}/mrpt/base/include/mrpt-eigen/Eigen/src/Core/arch/SSE/)
include_directories(${MRPT_PATH}/mrpt/base/include/mrpt-eigen/Eigen/src/Core/products/)
include_directories(${MRPT_PATH}/mrpt/base/include/mrpt-eigen/Eigen/src/Core/util/)
include_directories(${MRPT_PATH}/mrpt/base/include/mrpt-eigen/Eigen/src/Eigen2Support/)
include_directories(${MRPT_PATH}/mrpt/base/include/mrpt-eigen/Eigen/src/Eigen2Support/Geometry/)
include_directories(${MRPT_PATH}/mrpt/base/include/mrpt-eigen/Eigen/src/Eigenvalues/)
include_directories(${MRPT_PATH}/mrpt/base/include/mrpt-eigen/Eigen/src/Geometry/)
include_directories(${MRPT_PATH}/mrpt/base/include/mrpt-eigen/Eigen/src/Geometry/arch/)
include_directories(${MRPT_PATH}/mrpt/base/include/mrpt-eigen/Eigen/src/Householder/)
include_directories(${MRPT_PATH}/mrpt/base/include/mrpt-eigen/Eigen/src/IterativeLinearSolvers/)
include_directories(${MRPT_PATH}/mrpt/base/include/mrpt-eigen/Eigen/src/Jacobi/)
include_directories(${MRPT_PATH}/mrpt/base/include/mrpt-eigen/Eigen/src/LU/)
include_directories(${MRPT_PATH}/mrpt/base/include/mrpt-eigen/Eigen/src/LU/arch/)
include_directories(${MRPT_PATH}/mrpt/base/include/mrpt-eigen/Eigen/src/MetisSupport/)
include_directories(${MRPT_PATH}/mrpt/base/include/mrpt-eigen/Eigen/src/misc/)
include_directories(${MRPT_PATH}/mrpt/base/include/mrpt-eigen/Eigen/src/OrderingMethods/)
include_directories(${MRPT_PATH}/mrpt/base/include/mrpt-eigen/Eigen/src/PardisoSupport/)
include_directories(${MRPT_PATH}/mrpt/base/include/mrpt-eigen/Eigen/src/PaStiXSupport/)
include_directories(${MRPT_PATH}/mrpt/base/include/mrpt-eigen/Eigen/src/plugins/)
include_directories(${MRPT_PATH}/mrpt/base/include/mrpt-eigen/Eigen/src/QR/)
include_directories(${MRPT_PATH}/mrpt/base/include/mrpt-eigen/Eigen/src/SparseCholesky/)
include_directories(${MRPT_PATH}/mrpt/base/include/mrpt-eigen/Eigen/src/SparseCore/)
include_directories(${MRPT_PATH}/mrpt/base/include/mrpt-eigen/Eigen/src/SparseLU/)
include_directories(${MRPT_PATH}/mrpt/base/include/mrpt-eigen/Eigen/src/SparseQR/)
include_directories(${MRPT_PATH}/mrpt/base/include/mrpt-eigen/Eigen/src/SPQRSupport/)
include_directories(${MRPT_PATH}/mrpt/base/include/mrpt-eigen/Eigen/src/StlSupport/)
include_directories(${MRPT_PATH}/mrpt/base/include/mrpt-eigen/Eigen/src/SuperLUSupport/)
include_directories(${MRPT_PATH}/mrpt/base/include/mrpt-eigen/Eigen/src/SVD/)
include_directories(${MRPT_PATH}/mrpt/base/include/mrpt-eigen/Eigen/src/UmfPackSupport/)

include_directories(${MRPT_PATH}/mrpt/base/include/mrpt-eigen/unsupported/)
include_directories(${MRPT_PATH}/mrpt/base/include/mrpt-eigen/unsupported/bench/)
include_directories(${MRPT_PATH}/mrpt/base/include/mrpt-eigen/unsupported/doc/)
include_directories(${MRPT_PATH}/mrpt/base/include/mrpt-eigen/unsupported/doc/examples/)
include_directories(${MRPT_PATH}/mrpt/base/include/mrpt-eigen/unsupported/doc/snippets/)
include_directories(${MRPT_PATH}/mrpt/base/include/mrpt-eigen/unsupported/Eigen/)
include_directories(${MRPT_PATH}/mrpt/base/include/mrpt-eigen/unsupported/Eigen/src/)
include_directories(${MRPT_PATH}/mrpt/base/include/mrpt-eigen/unsupported/Eigen/src/AutoDiff/)
include_directories(${MRPT_PATH}/mrpt/base/include/mrpt-eigen/unsupported/Eigen/src/BVH/)
include_directories(${MRPT_PATH}/mrpt/base/include/mrpt-eigen/unsupported/Eigen/src/Eigenvalues/)
include_directories(${MRPT_PATH}/mrpt/base/include/mrpt-eigen/unsupported/Eigen/src/FFT/)
include_directories(${MRPT_PATH}/mrpt/base/include/mrpt-eigen/unsupported/Eigen/src/IterativeSolvers/)
include_directories(${MRPT_PATH}/mrpt/base/include/mrpt-eigen/unsupported/Eigen/src/KroneckerProduct/)
include_directories(${MRPT_PATH}/mrpt/base/include/mrpt-eigen/unsupported/Eigen/src/LevenbergMarquardt/)
include_directories(${MRPT_PATH}/mrpt/base/include/mrpt-eigen/unsupported/Eigen/src/MatrixFunctions/)
include_directories(${MRPT_PATH}/mrpt/base/include/mrpt-eigen/unsupported/Eigen/src/MoreVectorization/)
include_directories(${MRPT_PATH}/mrpt/base/include/mrpt-eigen/unsupported/Eigen/src/NonLinearOptimization/)
include_directories(${MRPT_PATH}/mrpt/base/include/mrpt-eigen/unsupported/Eigen/src/NumericalDiff/)
include_directories(${MRPT_PATH}/mrpt/base/include/mrpt-eigen/unsupported/Eigen/src/Polynomials/)
include_directories(${MRPT_PATH}/mrpt/base/include/mrpt-eigen/unsupported/Eigen/src/Skyline/)
include_directories(${MRPT_PATH}/mrpt/base/include/mrpt-eigen/unsupported/Eigen/src/SparseExtra/)
include_directories(${MRPT_PATH}/mrpt/base/include/mrpt-eigen/unsupported/Eigen/src/Splines/)
include_directories(${MRPT_PATH}/mrpt/base/include/mrpt-eigen/unsupported/Eigen/src/SVD/)
include_directories(${MRPT_PATH}/mrpt/base/include/mrpt-eigen/unsupported/Eigen/test/)
include_directories(${MRPT_PATH}/mrpt/base/include/mrpt-eigen/unsupported/Eigen/test/mpreal/)

############## mrpt_hwdrivers #################
include_directories(${MRPT_PATH}/mrpt/hwdrivers/)
include_directories(${MRPT_PATH}/mrpt/hwdrivers/include/)
include_directories(${MRPT_PATH}/mrpt/hwdrivers/include/mrpt/)
include_directories(${MRPT_PATH}/mrpt/hwdrivers/include/mrpt/hwdrivers/)

############## mrpt_maps #################
include_directories(${MRPT_PATH}/mrpt/maps/include/)
include_directories(${MRPT_PATH}/mrpt/maps/include/mrpt/)
include_directories(${MRPT_PATH}/mrpt/maps/include/mrpt/maps/)
include_directories(${MRPT_PATH}/mrpt/maps/include/mrpt/slam/)

############## marpt_obs #################
include_directories(${MRPT_PATH}/mrpt/obs/)
include_directories(${MRPT_PATH}/mrpt/obs/include/)
include_directories(${MRPT_PATH}/mrpt/obs/include/mrpt/)
include_directories(${MRPT_PATH}/mrpt/obs/include/mrpt/obs/)
include_directories(${MRPT_PATH}/mrpt/obs/include/mrpt/slam/)

############## marpt_scanmatching #################
include_directories(${MRPT_PATH}/mrpt/scanmatching/)
include_directories(${MRPT_PATH}/mrpt/scanmatching/include/)
include_directories(${MRPT_PATH}/mrpt/scanmatching/include/mrpt/)
include_directories(${MRPT_PATH}/mrpt/scanmatching/include/mrpt/scanmatching/)

############## mrpt_scanmatching #################
include_directories(${MRPT_PATH}/mrpt/slam/)
include_directories(${MRPT_PATH}/mrpt/slam/include/)
include_directories(${MRPT_PATH}/mrpt/slam/include/mrpt/)
include_directories(${MRPT_PATH}/mrpt/slam/include/mrpt/slam/)

SET(MRPT_LIBS
			${MRPT_PATH}/libmrpt-base.so
			${MRPT_PATH}/libmrpt-hwdrivers.so
			${MRPT_PATH}/libmrpt-maps.so
			${MRPT_PATH}/libmrpt-obs.so
			${MRPT_PATH}/libmrpt-scanmatching.so
			${MRPT_PATH}/libmrpt-slam.so
			# ${MRPT_PATH}/libplatform-utils.so
			)

set(CMAKE_INSTALL_RPATH "lib")

add_executable(RlApi_ecx
        src/main.cc
        #src/opencv_test.cc
        #src/deeplabV3_test.cc
        #src/CDeeplabV3Segamentation.cc
        )

set(Rl_API_PATH /home/xiang/Estudiar/AI/RL-SLAM/RlApi/lib/libRlApi.so)

# install target and libraries
set(CMAKE_INSTALL_PREFIX ${CMAKE_SOURCE_DIR}/install/RlApi)
install(TARGETS RlApi_ecx DESTINATION ./)
install(FILES model/ckpt.rknn DESTINATION ./)
# install(FILES model/dog_224x224.jpg DESTINATION ./)

install(PROGRAMS ${RKNN_API_LIB} DESTINATION lib)
install(PROGRAMS ${MRPT_LIBS} DESTINATION lib)

set (SRC_LIST ${PROJECT_SOURCE_DIR}/src/rl_api.cpp)
add_library (main_shared SHARED ${SRC_LIST})
#add_library (main_static STATIC ${SRC_LIST})

target_link_libraries(main_shared ${RKNN_API_LIB} ${MRPT_LIBS} pthread dl)


set_target_properties (main_shared PROPERTIES OUTPUT_NAME "RlApi")
#set_target_properties (main_static PROPERTIES OUTPUT_NAME "RlApi")

set (LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)

target_link_libraries(RlApi_ecx
	
	${MRPT_LIBS}
   ${Rl_API_PATH}
  pthread
  dl
)



set (EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
