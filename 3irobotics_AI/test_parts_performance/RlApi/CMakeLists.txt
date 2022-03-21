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
#set(CMAKE_C_COMPILER aarch64-linux-gnu-gcc)
#set(CMAKE_CXX_COMPILER aarch64-linux-gnu-g++)

SET(CMAKE_C_COMPILER "arm-linux-gnueabihf-gcc")
SET(CMAKE_CXX_COMPILER "arm-linux-gnueabihf-g++")


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
set(RKNN_API_PATH ${CMAKE_SOURCE_DIR}/librknn_api)
# include_directories(${RKNN_API_PATH}/include)
set(RKNN_API_LIB ${RKNN_API_PATH}/${LIB_ARCH}/librknn_api.so)

message(${RKNN_API_LIB})
# Rl api & rknn api
set(RL_API_PATH ${CMAKE_SOURCE_DIR}/include)
include_directories(${RL_API_PATH})

set(CMAKE_INSTALL_RPATH "lib")

#add_executable(RlApi_ecx
#        src/main.cc
        #src/opencv_test.cc
        #src/deeplabV3_test.cc
        #src/CDeeplabV3Segamentation.cc
#        )

# set(Rl_API_PATH /home/xiang/Estudiar/AI/RL-SLAM/RlApi/lib/libRlApi.so)

# install target and libraries
set(CMAKE_INSTALL_PREFIX ${CMAKE_SOURCE_DIR}/install/RlApi)
# install(TARGETS RlApi_ecx DESTINATION ./)
# install(FILES model/ckpt_precompile_20_rk161.rknn DESTINATION ./)
# install(FILES model/dog_224x224.jpg DESTINATION ./)

install(PROGRAMS ${RKNN_API_LIB} DESTINATION lib)
# install(PROGRAMS ${MRPT_LIBS} DESTINATION lib)

set (SRC_LIST ${PROJECT_SOURCE_DIR}/src/rl_api.cpp)
add_library (main_shared SHARED ${SRC_LIST})
#add_library (main_static STATIC ${SRC_LIST})

target_link_libraries(main_shared ${RKNN_API_LIB} pthread dl)


set_target_properties (main_shared PROPERTIES OUTPUT_NAME "RlApi")
#set_target_properties (main_static PROPERTIES OUTPUT_NAME "RlApi")

set (LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)

#target_link_libraries(RlApi_ecx
# ${MRPT_LIBS}
#  ${Rl_API_PATH}
#  pthread
#  dl
#)

# set (EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
