cmake_minimum_required(VERSION 3.4.1)

project(rknn_class_infer_1126_linux_scene)

set(CMAKE_SYSTEM_NAME Linux)
add_definitions(-w)
# for aarch64
#set(CROSS_COMPILE_TOOL ${CMAKE_SOURCE_DIR}/../../../../../../prebuilts/gcc/linux-x86/aarch64/gcc-linaro-6.3.1-2017.05-x86_64_aarch64-linux-gnu/bin/aarch64-linux-gnu)

# for armhf
# set(CROSS_COMPILE_TOOL ${CMAKE_SOURCE_DIR}/../../../../../../prebuilts/gcc/linux-x86/arm/gcc-linaro-6.3.1-2017.05-x86_64_arm-linux-gnueabihf/bin/arm-linux-gnueabihf)

#1126
# set(CROSS_COMPILE_TOOL /home/zkj/Documents/Windows_Share/Windows_share/1126-cross-compile-tool/gcc-arm-linux-gnueabihf/bin/arm-linux-gnueabihf)
set(CROSS_COMPILE_TOOL /home/agent/gcc-arm-8.3-2019.03-x86_64-arm-linux-gnueabihf/bin/arm-linux-gnueabihf)
set(CMAKE_C_COMPILER ${CROSS_COMPILE_TOOL}-gcc)
set(CMAKE_CXX_COMPILER ${CROSS_COMPILE_TOOL}-g++)

#1806/1808
# SET(TOOLCHAIN_LIB /home/agent/aarch64-rockchip-linux-toolchain/)
# set(CMAKE_C_COMPILER aarch64-linux-gnu-gcc)
# set(CMAKE_CXX_COMPILER aarch64-linux-gnu-g++)


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

# eigen
find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIRS})

INCLUDE_DIRECTORIES(include)

# rknn api
set(RKNN_API_PATH ${CMAKE_SOURCE_DIR}/../../librknn_api)
include_directories(${RKNN_API_PATH}/include)
set(RKNN_API_LIB ${RKNN_API_PATH}/${LIB_ARCH}/librknn_api.so)

# opencv
set(OPENCV_PATH ${CMAKE_SOURCE_DIR}/../libs/opencv/opencv410_arm32)
include_directories(${OPENCV_PATH}/include)
include_directories(${OPENCV_PATH}/include/opencv4/)
include_directories(${OPENCV_PATH}/include/opencv4/opencv2)
#file(GLOB OPENCV_LIBS ${OPENCV_PATH}/${LIB_ARCH}/*.so*)
#file(GLOB OPENCV_LIBS ${OPENCV_PATH}/lib/*.so*)
#file(GLOB OPENCV_STATIC_LIBS ${OPENCV_PATH}/lib/*.so*)

SET(OPENCV_LIBS
			${OPENCV_PATH}/lib/libopencv_world.so
			${OPENCV_PATH}/lib/opencv4/3rdparty/liblibjpeg-turbo.a
			${OPENCV_PATH}/lib/opencv4/3rdparty/libade.a
			${OPENCV_PATH}/lib/opencv4/3rdparty/libIlmImf.a
			${OPENCV_PATH}/lib/opencv4/3rdparty/liblibjasper.a
			${OPENCV_PATH}/lib/opencv4/3rdparty/liblibjpeg-turbo.a
			${OPENCV_PATH}/lib/opencv4/3rdparty/liblibtiff.a
			${OPENCV_PATH}/lib/opencv4/3rdparty/liblibwebp.a
			${OPENCV_PATH}/lib/opencv4/3rdparty/libquirc.a 
			${OPENCV_PATH}/lib/opencv4/3rdparty/libzlib.a
			)

set(CMAKE_INSTALL_RPATH "lib")

add_executable(rknn_class_infer_1126_linux_scene
        #src/main_M.cc
        src/main.cc
		src/SceneNet.cpp
        )

target_link_libraries(rknn_class_infer_1126_linux_scene
	${RKNN_API_LIB}
	${OPENCV_LIBS}
  pthread
  dl
)


# install target and libraries
set(CMAKE_INSTALL_PREFIX ${CMAKE_SOURCE_DIR}/install/rknn_class_infer_1126_linux_scene)
install(TARGETS rknn_class_infer_1126_linux_scene DESTINATION ./)
install(FILES model/SceneResnet18_18pth.rknn DESTINATION ./)
install(FILES model/test_toilet.jpg DESTINATION ./)

install(PROGRAMS ${RKNN_API_LIB} DESTINATION lib)
install(PROGRAMS ${OPENCV_LIBS} DESTINATION lib)


# SET(SRC_LIST src/main.cc)
# # set(RKNN_API_LIB /home/agent/C-_Project/Test_Vs_Cmake/struct_test/librknn_api/lib64/librknn_api.so)

# add_executable (test ${SRC_LIST})

# # after ADD_EXECUTABLE，为生成文件target添加库，这个必须写在ADD_EXECUTABLE()之后，为生成文件target添加库。
# # TARGET_LINK_LIBRARIES(target library1 <debug | optimized> library2...)
# target_link_libraries(test ${OpenCV_LIBS} ${RKNN_API_LIB})

# set (EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)