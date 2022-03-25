# project name
PROJECT(Scene)

# using C++11  环境
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x -O3 -mtune=native ")

# cmake最低版本
CMAKE_MINIMUM_REQUIRED(VERSION 3.10)


# eigen
find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIRS})
# include_directories(/home/agent/aarch64-rockchip-linux-toolchain/3rdParty/include/eigen3)
INCLUDE_DIRECTORIES(include)

# rknn api
set(RKNN_API_PATH ${CMAKE_SOURCE_DIR}/../../librknn_api)
include_directories(${RKNN_API_PATH}/include)
set(RKNN_API_LIB /home/agent/rk_test_project/librknn_api/lib64/librknn_api.so)


# opencv 1808/6
set(OPENCV_PATH ${CMAKE_SOURCE_DIR}/../libs/opencv/opencv410_aarch64)
include_directories(${OPENCV_PATH}/include)
file(GLOB OPENCV_LIBS /home/agent/rk_test_project/examples/libs/opencv/opencv410_aarch64/lib64/*.so*)



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
