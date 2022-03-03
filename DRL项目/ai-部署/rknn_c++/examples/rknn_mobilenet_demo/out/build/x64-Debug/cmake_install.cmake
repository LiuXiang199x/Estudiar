# Install script for directory: C:/Users/Administrator/Desktop/3irobotics/ai-部署/rknn_c++/examples/rknn_mobilenet_demo

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "C:/Users/Administrator/Desktop/3irobotics/ai-部署/rknn_c++/examples/rknn_mobilenet_demo/install/rknn_class_infer")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "Debug")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "FALSE")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/." TYPE EXECUTABLE FILES "C:/Users/Administrator/Desktop/3irobotics/ai-部署/rknn_c++/examples/rknn_mobilenet_demo/out/build/x64-Debug/rknn_class_infer.exe")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/." TYPE FILE FILES "C:/Users/Administrator/Desktop/3irobotics/ai-部署/rknn_c++/examples/rknn_mobilenet_demo/model/mobilenet_v1.rknn")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/." TYPE FILE FILES "C:/Users/Administrator/Desktop/3irobotics/ai-部署/rknn_c++/examples/rknn_mobilenet_demo/model/dog_224x224.jpg")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE PROGRAM FILES "C:/Users/Administrator/Desktop/3irobotics/ai-部署/rknn_c++/examples/rknn_mobilenet_demo/../../librknn_api/lib/librknn_api.so")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE PROGRAM FILES
    "C:/Users/Administrator/Desktop/3irobotics/ai-部署/rknn_c++/examples/rknn_mobilenet_demo/../libs/opencv/opencv410/lib/libopencv_world.so"
    "C:/Users/Administrator/Desktop/3irobotics/ai-部署/rknn_c++/examples/rknn_mobilenet_demo/../libs/opencv/opencv410/lib/opencv4/3rdparty/liblibjpeg-turbo.a"
    "C:/Users/Administrator/Desktop/3irobotics/ai-部署/rknn_c++/examples/rknn_mobilenet_demo/../libs/opencv/opencv410/lib/opencv4/3rdparty/libade.a"
    "C:/Users/Administrator/Desktop/3irobotics/ai-部署/rknn_c++/examples/rknn_mobilenet_demo/../libs/opencv/opencv410/lib/opencv4/3rdparty/libIlmImf.a"
    "C:/Users/Administrator/Desktop/3irobotics/ai-部署/rknn_c++/examples/rknn_mobilenet_demo/../libs/opencv/opencv410/lib/opencv4/3rdparty/liblibjasper.a"
    "C:/Users/Administrator/Desktop/3irobotics/ai-部署/rknn_c++/examples/rknn_mobilenet_demo/../libs/opencv/opencv410/lib/opencv4/3rdparty/liblibjpeg-turbo.a"
    "C:/Users/Administrator/Desktop/3irobotics/ai-部署/rknn_c++/examples/rknn_mobilenet_demo/../libs/opencv/opencv410/lib/opencv4/3rdparty/liblibtiff.a"
    "C:/Users/Administrator/Desktop/3irobotics/ai-部署/rknn_c++/examples/rknn_mobilenet_demo/../libs/opencv/opencv410/lib/opencv4/3rdparty/liblibwebp.a"
    "C:/Users/Administrator/Desktop/3irobotics/ai-部署/rknn_c++/examples/rknn_mobilenet_demo/../libs/opencv/opencv410/lib/opencv4/3rdparty/libquirc.a"
    "C:/Users/Administrator/Desktop/3irobotics/ai-部署/rknn_c++/examples/rknn_mobilenet_demo/../libs/opencv/opencv410/lib/opencv4/3rdparty/libzlib.a"
    )
endif()

if(CMAKE_INSTALL_COMPONENT)
  set(CMAKE_INSTALL_MANIFEST "install_manifest_${CMAKE_INSTALL_COMPONENT}.txt")
else()
  set(CMAKE_INSTALL_MANIFEST "install_manifest.txt")
endif()

string(REPLACE ";" "\n" CMAKE_INSTALL_MANIFEST_CONTENT
       "${CMAKE_INSTALL_MANIFEST_FILES}")
file(WRITE "C:/Users/Administrator/Desktop/3irobotics/ai-部署/rknn_c++/examples/rknn_mobilenet_demo/out/build/x64-Debug/${CMAKE_INSTALL_MANIFEST}"
     "${CMAKE_INSTALL_MANIFEST_CONTENT}")
