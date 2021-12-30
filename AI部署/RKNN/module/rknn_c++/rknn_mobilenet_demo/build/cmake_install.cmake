# Install script for directory: /home/xiang/rknn_c++/rknn_mobilenet_demo

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/home/xiang/rknn_c++/rknn_mobilenet_demo/install/rknn_class_infer")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "")
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

# Install shared libraries without execute permission?
if(NOT DEFINED CMAKE_INSTALL_SO_NO_EXE)
  set(CMAKE_INSTALL_SO_NO_EXE "1")
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "FALSE")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/./rknn_class_infer" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/./rknn_class_infer")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/./rknn_class_infer"
         RPATH "lib")
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/." TYPE EXECUTABLE FILES "/home/xiang/rknn_c++/rknn_mobilenet_demo/build/rknn_class_infer")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/./rknn_class_infer" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/./rknn_class_infer")
    file(RPATH_CHANGE
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/./rknn_class_infer"
         OLD_RPATH "/home/xiang/rknn_c++/rknn_mobilenet_demo/../librknn_api/lib:/home/xiang/rknn_c++/rknn_mobilenet_demo/../libs/opencv/opencv410/lib:"
         NEW_RPATH "lib")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/./rknn_class_infer")
    endif()
  endif()
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/." TYPE FILE FILES "/home/xiang/rknn_c++/rknn_mobilenet_demo/model/mobilenet_v1.rknn")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/." TYPE FILE FILES "/home/xiang/rknn_c++/rknn_mobilenet_demo/model/dog_224x224.jpg")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE PROGRAM FILES "/home/xiang/rknn_c++/rknn_mobilenet_demo/../librknn_api/lib/librknn_api.so")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE PROGRAM FILES
    "/home/xiang/rknn_c++/rknn_mobilenet_demo/../libs/opencv/opencv410/lib/libopencv_world.so"
    "/home/xiang/rknn_c++/rknn_mobilenet_demo/../libs/opencv/opencv410/lib/opencv4/3rdparty/liblibjpeg-turbo.a"
    "/home/xiang/rknn_c++/rknn_mobilenet_demo/../libs/opencv/opencv410/lib/opencv4/3rdparty/libade.a"
    "/home/xiang/rknn_c++/rknn_mobilenet_demo/../libs/opencv/opencv410/lib/opencv4/3rdparty/libIlmImf.a"
    "/home/xiang/rknn_c++/rknn_mobilenet_demo/../libs/opencv/opencv410/lib/opencv4/3rdparty/liblibjasper.a"
    "/home/xiang/rknn_c++/rknn_mobilenet_demo/../libs/opencv/opencv410/lib/opencv4/3rdparty/liblibjpeg-turbo.a"
    "/home/xiang/rknn_c++/rknn_mobilenet_demo/../libs/opencv/opencv410/lib/opencv4/3rdparty/liblibtiff.a"
    "/home/xiang/rknn_c++/rknn_mobilenet_demo/../libs/opencv/opencv410/lib/opencv4/3rdparty/liblibwebp.a"
    "/home/xiang/rknn_c++/rknn_mobilenet_demo/../libs/opencv/opencv410/lib/opencv4/3rdparty/libquirc.a"
    "/home/xiang/rknn_c++/rknn_mobilenet_demo/../libs/opencv/opencv410/lib/opencv4/3rdparty/libzlib.a"
    )
endif()

if(CMAKE_INSTALL_COMPONENT)
  set(CMAKE_INSTALL_MANIFEST "install_manifest_${CMAKE_INSTALL_COMPONENT}.txt")
else()
  set(CMAKE_INSTALL_MANIFEST "install_manifest.txt")
endif()

string(REPLACE ";" "\n" CMAKE_INSTALL_MANIFEST_CONTENT
       "${CMAKE_INSTALL_MANIFEST_FILES}")
file(WRITE "/home/xiang/rknn_c++/rknn_mobilenet_demo/build/${CMAKE_INSTALL_MANIFEST}"
     "${CMAKE_INSTALL_MANIFEST_CONTENT}")
