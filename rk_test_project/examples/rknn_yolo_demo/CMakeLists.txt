cmake_minimum_required(VERSION 3.4.1)

project(rknn_yolo_demo_linux)

set(CMAKE_SYSTEM_NAME Linux)

# for aarch64
set(CROSS_COMPILE_TOOL ${CMAKE_SOURCE_DIR}/../../../../../../prebuilts/gcc/linux-x86/aarch64/gcc-linaro-6.3.1-2017.05-x86_64_aarch64-linux-gnu/bin/aarch64-linux-gnu)

# for armhf
# set(CROSS_COMPILE_TOOL ${CMAKE_SOURCE_DIR}/../../../../../../prebuilts/gcc/linux-x86/arm/gcc-linaro-6.3.1-2017.05-x86_64_arm-linux-gnueabihf/bin/arm-linux-gnueabihf)

#set(CMAKE_C_COMPILER ${CROSS_COMPILE_TOOL}-gcc)
#set(CMAKE_CXX_COMPILER ${CROSS_COMPILE_TOOL}-g++)
#1806/1808
SET(TOOLCHAIN_LIB /opt/aarch64-rockchip-linux-toolchain/)
SET(CMAKE_C_COMPILER "aarch64-linux-gcc")
SET(CMAKE_CXX_COMPILER "aarch64-linux-g++")

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

if (CMAKE_C_COMPILER MATCHES "aarch64")
  set(LIB_ARCH lib64)
else()
  set(LIB_ARCH lib)
endif()

# rknn api
set(RKNN_API_PATH ${CMAKE_SOURCE_DIR}/../../librknn_api)
include_directories(${RKNN_API_PATH}/include)
set(RKNN_API_LIB ${RKNN_API_PATH}/${LIB_ARCH}/librknn_api.so)

# opencv-1806
set(OPENCV_PATH ${CMAKE_SOURCE_DIR}/../libs/opencv/opencv410_aarch64)
include_directories(${OPENCV_PATH}/include)
file(GLOB OPENCV_LIBS ${OPENCV_PATH}/${LIB_ARCH}/*.so*)

set(CMAKE_INSTALL_RPATH "lib")

add_executable(rknn_yolo_demo
        #src/main_M.cc
        src/SCYolo3Detecor.cc
        src/SCYolov3_post_process.cc
        #src/SCYolov3_post_process_3head.cc
        src/test_yolo.cc
        )

target_link_libraries(rknn_yolo_demo
	${RKNN_API_LIB}
	${OPENCV_LIBS}
)

# install target and libraries
set(CMAKE_INSTALL_PREFIX ${CMAKE_SOURCE_DIR}/install/rknn_yolo_demo)
install(TARGETS rknn_yolo_demo DESTINATION ./)
install(FILES model/yolov3.rknn DESTINATION ./)
install(FILES model/dog.jpg DESTINATION ./)
install(PROGRAMS ${RKNN_API_LIB} DESTINATION lib)
install(PROGRAMS ${OPENCV_LIBS} DESTINATION lib)
