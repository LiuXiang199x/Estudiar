# Install script for directory: /home/agent/rk_test_project/examples/sceneClass

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/home/agent/rk_test_project/examples/sceneClass/install/rknn_class_infer_1126_linux_scene")
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
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/./rknn_class_infer_1126_linux_scene" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/./rknn_class_infer_1126_linux_scene")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/./rknn_class_infer_1126_linux_scene"
         RPATH "lib")
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/." TYPE EXECUTABLE FILES "/home/agent/rk_test_project/examples/sceneClass/build/rknn_class_infer_1126_linux_scene")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/./rknn_class_infer_1126_linux_scene" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/./rknn_class_infer_1126_linux_scene")
    file(RPATH_CHANGE
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/./rknn_class_infer_1126_linux_scene"
         OLD_RPATH "/home/agent/rk_test_project/librknn_api/lib64:/home/agent/rk_test_project/examples/libs/opencv/opencv410_aarch64/lib64:"
         NEW_RPATH "lib")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/./rknn_class_infer_1126_linux_scene")
    endif()
  endif()
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/." TYPE FILE FILES "/home/agent/rk_test_project/examples/sceneClass/model/SceneResnet18_18pth.rknn")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/." TYPE FILE FILES "/home/agent/rk_test_project/examples/sceneClass/model/test_toilet.jpg")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE PROGRAM FILES "/home/agent/rk_test_project/librknn_api/lib64/librknn_api.so")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE PROGRAM FILES
    "/home/agent/rk_test_project/examples/libs/opencv/opencv410_aarch64/lib64/libopencv_calib3d.so"
    "/home/agent/rk_test_project/examples/libs/opencv/opencv410_aarch64/lib64/libopencv_calib3d.so.4.1"
    "/home/agent/rk_test_project/examples/libs/opencv/opencv410_aarch64/lib64/libopencv_calib3d.so.4.1.0"
    "/home/agent/rk_test_project/examples/libs/opencv/opencv410_aarch64/lib64/libopencv_core.so"
    "/home/agent/rk_test_project/examples/libs/opencv/opencv410_aarch64/lib64/libopencv_core.so.4.1"
    "/home/agent/rk_test_project/examples/libs/opencv/opencv410_aarch64/lib64/libopencv_core.so.4.1.0"
    "/home/agent/rk_test_project/examples/libs/opencv/opencv410_aarch64/lib64/libopencv_dnn.so"
    "/home/agent/rk_test_project/examples/libs/opencv/opencv410_aarch64/lib64/libopencv_dnn.so.4.1"
    "/home/agent/rk_test_project/examples/libs/opencv/opencv410_aarch64/lib64/libopencv_dnn.so.4.1.0"
    "/home/agent/rk_test_project/examples/libs/opencv/opencv410_aarch64/lib64/libopencv_features2d.so"
    "/home/agent/rk_test_project/examples/libs/opencv/opencv410_aarch64/lib64/libopencv_features2d.so.4.1"
    "/home/agent/rk_test_project/examples/libs/opencv/opencv410_aarch64/lib64/libopencv_features2d.so.4.1.0"
    "/home/agent/rk_test_project/examples/libs/opencv/opencv410_aarch64/lib64/libopencv_flann.so"
    "/home/agent/rk_test_project/examples/libs/opencv/opencv410_aarch64/lib64/libopencv_flann.so.4.1"
    "/home/agent/rk_test_project/examples/libs/opencv/opencv410_aarch64/lib64/libopencv_flann.so.4.1.0"
    "/home/agent/rk_test_project/examples/libs/opencv/opencv410_aarch64/lib64/libopencv_gapi.so"
    "/home/agent/rk_test_project/examples/libs/opencv/opencv410_aarch64/lib64/libopencv_gapi.so.4.1"
    "/home/agent/rk_test_project/examples/libs/opencv/opencv410_aarch64/lib64/libopencv_gapi.so.4.1.0"
    "/home/agent/rk_test_project/examples/libs/opencv/opencv410_aarch64/lib64/libopencv_highgui.so"
    "/home/agent/rk_test_project/examples/libs/opencv/opencv410_aarch64/lib64/libopencv_highgui.so.4.1"
    "/home/agent/rk_test_project/examples/libs/opencv/opencv410_aarch64/lib64/libopencv_highgui.so.4.1.0"
    "/home/agent/rk_test_project/examples/libs/opencv/opencv410_aarch64/lib64/libopencv_imgcodecs.so"
    "/home/agent/rk_test_project/examples/libs/opencv/opencv410_aarch64/lib64/libopencv_imgcodecs.so.4.1"
    "/home/agent/rk_test_project/examples/libs/opencv/opencv410_aarch64/lib64/libopencv_imgcodecs.so.4.1.0"
    "/home/agent/rk_test_project/examples/libs/opencv/opencv410_aarch64/lib64/libopencv_imgproc.so"
    "/home/agent/rk_test_project/examples/libs/opencv/opencv410_aarch64/lib64/libopencv_imgproc.so.4.1"
    "/home/agent/rk_test_project/examples/libs/opencv/opencv410_aarch64/lib64/libopencv_imgproc.so.4.1.0"
    "/home/agent/rk_test_project/examples/libs/opencv/opencv410_aarch64/lib64/libopencv_ml.so"
    "/home/agent/rk_test_project/examples/libs/opencv/opencv410_aarch64/lib64/libopencv_ml.so.4.1"
    "/home/agent/rk_test_project/examples/libs/opencv/opencv410_aarch64/lib64/libopencv_ml.so.4.1.0"
    "/home/agent/rk_test_project/examples/libs/opencv/opencv410_aarch64/lib64/libopencv_objdetect.so"
    "/home/agent/rk_test_project/examples/libs/opencv/opencv410_aarch64/lib64/libopencv_objdetect.so.4.1"
    "/home/agent/rk_test_project/examples/libs/opencv/opencv410_aarch64/lib64/libopencv_objdetect.so.4.1.0"
    "/home/agent/rk_test_project/examples/libs/opencv/opencv410_aarch64/lib64/libopencv_photo.so"
    "/home/agent/rk_test_project/examples/libs/opencv/opencv410_aarch64/lib64/libopencv_photo.so.4.1"
    "/home/agent/rk_test_project/examples/libs/opencv/opencv410_aarch64/lib64/libopencv_photo.so.4.1.0"
    "/home/agent/rk_test_project/examples/libs/opencv/opencv410_aarch64/lib64/libopencv_stitching.so"
    "/home/agent/rk_test_project/examples/libs/opencv/opencv410_aarch64/lib64/libopencv_stitching.so.4.1"
    "/home/agent/rk_test_project/examples/libs/opencv/opencv410_aarch64/lib64/libopencv_stitching.so.4.1.0"
    "/home/agent/rk_test_project/examples/libs/opencv/opencv410_aarch64/lib64/libopencv_video.so"
    "/home/agent/rk_test_project/examples/libs/opencv/opencv410_aarch64/lib64/libopencv_video.so.4.1"
    "/home/agent/rk_test_project/examples/libs/opencv/opencv410_aarch64/lib64/libopencv_video.so.4.1.0"
    "/home/agent/rk_test_project/examples/libs/opencv/opencv410_aarch64/lib64/libopencv_videoio.so"
    "/home/agent/rk_test_project/examples/libs/opencv/opencv410_aarch64/lib64/libopencv_videoio.so.4.1"
    "/home/agent/rk_test_project/examples/libs/opencv/opencv410_aarch64/lib64/libopencv_videoio.so.4.1.0"
    "/home/agent/rk_test_project/examples/libs/opencv/opencv410_aarch64/lib64/libz.so"
    "/home/agent/rk_test_project/examples/libs/opencv/opencv410_aarch64/lib64/libz.so.1"
    "/home/agent/rk_test_project/examples/libs/opencv/opencv410_aarch64/lib64/libz.so.1.2.11"
    )
endif()

if(CMAKE_INSTALL_COMPONENT)
  set(CMAKE_INSTALL_MANIFEST "install_manifest_${CMAKE_INSTALL_COMPONENT}.txt")
else()
  set(CMAKE_INSTALL_MANIFEST "install_manifest.txt")
endif()

string(REPLACE ";" "\n" CMAKE_INSTALL_MANIFEST_CONTENT
       "${CMAKE_INSTALL_MANIFEST_FILES}")
file(WRITE "/home/agent/rk_test_project/examples/sceneClass/build/${CMAKE_INSTALL_MANIFEST}"
     "${CMAKE_INSTALL_MANIFEST_CONTENT}")
