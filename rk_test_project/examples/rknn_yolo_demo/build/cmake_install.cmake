# Install script for directory: /home/zkj/Documents/Windows_Share/Windows_share_27F4/Board_workspace/rknn_board_test/rk_test_project/examples/rknn_yolo_demo

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/home/zkj/Documents/Windows_Share/Windows_share_27F4/Board_workspace/rknn_board_test/rk_test_project/examples/rknn_yolo_demo/install/rknn_yolo_demo")
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

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/./rknn_yolo_demo" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/./rknn_yolo_demo")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/./rknn_yolo_demo"
         RPATH "lib")
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/." TYPE EXECUTABLE FILES "/home/zkj/Documents/Windows_Share/Windows_share_27F4/Board_workspace/rknn_board_test/rk_test_project/examples/rknn_yolo_demo/build/rknn_yolo_demo")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/./rknn_yolo_demo" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/./rknn_yolo_demo")
    file(RPATH_CHANGE
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/./rknn_yolo_demo"
         OLD_RPATH "/home/zkj/Documents/Windows_Share/Windows_share_27F4/Board_workspace/rknn_board_test/rk_test_project/examples/rknn_yolo_demo/../../librknn_api/lib64:/home/zkj/Documents/Windows_Share/Windows_share_27F4/Board_workspace/rknn_board_test/rk_test_project/examples/rknn_yolo_demo/../libs/opencv/opencv410_aarch64/lib64:"
         NEW_RPATH "lib")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/./rknn_yolo_demo")
    endif()
  endif()
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/." TYPE FILE FILES "/home/zkj/Documents/Windows_Share/Windows_share_27F4/Board_workspace/rknn_board_test/rk_test_project/examples/rknn_yolo_demo/model/yolov3.rknn")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/." TYPE FILE FILES "/home/zkj/Documents/Windows_Share/Windows_share_27F4/Board_workspace/rknn_board_test/rk_test_project/examples/rknn_yolo_demo/model/dog.jpg")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE PROGRAM FILES "/home/zkj/Documents/Windows_Share/Windows_share_27F4/Board_workspace/rknn_board_test/rk_test_project/examples/rknn_yolo_demo/../../librknn_api/lib64/librknn_api.so")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE PROGRAM FILES
    "/home/zkj/Documents/Windows_Share/Windows_share_27F4/Board_workspace/rknn_board_test/rk_test_project/examples/rknn_yolo_demo/../libs/opencv/opencv410_aarch64/lib64/libopencv_imgproc.so.4.1.0"
    "/home/zkj/Documents/Windows_Share/Windows_share_27F4/Board_workspace/rknn_board_test/rk_test_project/examples/rknn_yolo_demo/../libs/opencv/opencv410_aarch64/lib64/libopencv_calib3d.so"
    "/home/zkj/Documents/Windows_Share/Windows_share_27F4/Board_workspace/rknn_board_test/rk_test_project/examples/rknn_yolo_demo/../libs/opencv/opencv410_aarch64/lib64/libopencv_calib3d.so.4.1"
    "/home/zkj/Documents/Windows_Share/Windows_share_27F4/Board_workspace/rknn_board_test/rk_test_project/examples/rknn_yolo_demo/../libs/opencv/opencv410_aarch64/lib64/libopencv_calib3d.so.4.1.0"
    "/home/zkj/Documents/Windows_Share/Windows_share_27F4/Board_workspace/rknn_board_test/rk_test_project/examples/rknn_yolo_demo/../libs/opencv/opencv410_aarch64/lib64/libopencv_core.so"
    "/home/zkj/Documents/Windows_Share/Windows_share_27F4/Board_workspace/rknn_board_test/rk_test_project/examples/rknn_yolo_demo/../libs/opencv/opencv410_aarch64/lib64/libopencv_core.so.4.1"
    "/home/zkj/Documents/Windows_Share/Windows_share_27F4/Board_workspace/rknn_board_test/rk_test_project/examples/rknn_yolo_demo/../libs/opencv/opencv410_aarch64/lib64/libopencv_core.so.4.1.0"
    "/home/zkj/Documents/Windows_Share/Windows_share_27F4/Board_workspace/rknn_board_test/rk_test_project/examples/rknn_yolo_demo/../libs/opencv/opencv410_aarch64/lib64/libopencv_dnn.so"
    "/home/zkj/Documents/Windows_Share/Windows_share_27F4/Board_workspace/rknn_board_test/rk_test_project/examples/rknn_yolo_demo/../libs/opencv/opencv410_aarch64/lib64/libopencv_dnn.so.4.1"
    "/home/zkj/Documents/Windows_Share/Windows_share_27F4/Board_workspace/rknn_board_test/rk_test_project/examples/rknn_yolo_demo/../libs/opencv/opencv410_aarch64/lib64/libopencv_dnn.so.4.1.0"
    "/home/zkj/Documents/Windows_Share/Windows_share_27F4/Board_workspace/rknn_board_test/rk_test_project/examples/rknn_yolo_demo/../libs/opencv/opencv410_aarch64/lib64/libopencv_features2d.so"
    "/home/zkj/Documents/Windows_Share/Windows_share_27F4/Board_workspace/rknn_board_test/rk_test_project/examples/rknn_yolo_demo/../libs/opencv/opencv410_aarch64/lib64/libopencv_features2d.so.4.1"
    "/home/zkj/Documents/Windows_Share/Windows_share_27F4/Board_workspace/rknn_board_test/rk_test_project/examples/rknn_yolo_demo/../libs/opencv/opencv410_aarch64/lib64/libopencv_features2d.so.4.1.0"
    "/home/zkj/Documents/Windows_Share/Windows_share_27F4/Board_workspace/rknn_board_test/rk_test_project/examples/rknn_yolo_demo/../libs/opencv/opencv410_aarch64/lib64/libopencv_flann.so"
    "/home/zkj/Documents/Windows_Share/Windows_share_27F4/Board_workspace/rknn_board_test/rk_test_project/examples/rknn_yolo_demo/../libs/opencv/opencv410_aarch64/lib64/libopencv_flann.so.4.1"
    "/home/zkj/Documents/Windows_Share/Windows_share_27F4/Board_workspace/rknn_board_test/rk_test_project/examples/rknn_yolo_demo/../libs/opencv/opencv410_aarch64/lib64/libopencv_flann.so.4.1.0"
    "/home/zkj/Documents/Windows_Share/Windows_share_27F4/Board_workspace/rknn_board_test/rk_test_project/examples/rknn_yolo_demo/../libs/opencv/opencv410_aarch64/lib64/libopencv_gapi.so"
    "/home/zkj/Documents/Windows_Share/Windows_share_27F4/Board_workspace/rknn_board_test/rk_test_project/examples/rknn_yolo_demo/../libs/opencv/opencv410_aarch64/lib64/libopencv_gapi.so.4.1"
    "/home/zkj/Documents/Windows_Share/Windows_share_27F4/Board_workspace/rknn_board_test/rk_test_project/examples/rknn_yolo_demo/../libs/opencv/opencv410_aarch64/lib64/libopencv_gapi.so.4.1.0"
    "/home/zkj/Documents/Windows_Share/Windows_share_27F4/Board_workspace/rknn_board_test/rk_test_project/examples/rknn_yolo_demo/../libs/opencv/opencv410_aarch64/lib64/libopencv_highgui.so"
    "/home/zkj/Documents/Windows_Share/Windows_share_27F4/Board_workspace/rknn_board_test/rk_test_project/examples/rknn_yolo_demo/../libs/opencv/opencv410_aarch64/lib64/libopencv_highgui.so.4.1"
    "/home/zkj/Documents/Windows_Share/Windows_share_27F4/Board_workspace/rknn_board_test/rk_test_project/examples/rknn_yolo_demo/../libs/opencv/opencv410_aarch64/lib64/libopencv_highgui.so.4.1.0"
    "/home/zkj/Documents/Windows_Share/Windows_share_27F4/Board_workspace/rknn_board_test/rk_test_project/examples/rknn_yolo_demo/../libs/opencv/opencv410_aarch64/lib64/libopencv_imgcodecs.so"
    "/home/zkj/Documents/Windows_Share/Windows_share_27F4/Board_workspace/rknn_board_test/rk_test_project/examples/rknn_yolo_demo/../libs/opencv/opencv410_aarch64/lib64/libopencv_imgcodecs.so.4.1"
    "/home/zkj/Documents/Windows_Share/Windows_share_27F4/Board_workspace/rknn_board_test/rk_test_project/examples/rknn_yolo_demo/../libs/opencv/opencv410_aarch64/lib64/libopencv_imgcodecs.so.4.1.0"
    "/home/zkj/Documents/Windows_Share/Windows_share_27F4/Board_workspace/rknn_board_test/rk_test_project/examples/rknn_yolo_demo/../libs/opencv/opencv410_aarch64/lib64/libopencv_imgproc.so"
    "/home/zkj/Documents/Windows_Share/Windows_share_27F4/Board_workspace/rknn_board_test/rk_test_project/examples/rknn_yolo_demo/../libs/opencv/opencv410_aarch64/lib64/libopencv_imgproc.so.4.1"
    "/home/zkj/Documents/Windows_Share/Windows_share_27F4/Board_workspace/rknn_board_test/rk_test_project/examples/rknn_yolo_demo/../libs/opencv/opencv410_aarch64/lib64/libopencv_ml.so"
    "/home/zkj/Documents/Windows_Share/Windows_share_27F4/Board_workspace/rknn_board_test/rk_test_project/examples/rknn_yolo_demo/../libs/opencv/opencv410_aarch64/lib64/libopencv_ml.so.4.1"
    "/home/zkj/Documents/Windows_Share/Windows_share_27F4/Board_workspace/rknn_board_test/rk_test_project/examples/rknn_yolo_demo/../libs/opencv/opencv410_aarch64/lib64/libopencv_ml.so.4.1.0"
    "/home/zkj/Documents/Windows_Share/Windows_share_27F4/Board_workspace/rknn_board_test/rk_test_project/examples/rknn_yolo_demo/../libs/opencv/opencv410_aarch64/lib64/libopencv_objdetect.so"
    "/home/zkj/Documents/Windows_Share/Windows_share_27F4/Board_workspace/rknn_board_test/rk_test_project/examples/rknn_yolo_demo/../libs/opencv/opencv410_aarch64/lib64/libopencv_objdetect.so.4.1"
    "/home/zkj/Documents/Windows_Share/Windows_share_27F4/Board_workspace/rknn_board_test/rk_test_project/examples/rknn_yolo_demo/../libs/opencv/opencv410_aarch64/lib64/libopencv_objdetect.so.4.1.0"
    "/home/zkj/Documents/Windows_Share/Windows_share_27F4/Board_workspace/rknn_board_test/rk_test_project/examples/rknn_yolo_demo/../libs/opencv/opencv410_aarch64/lib64/libopencv_photo.so"
    "/home/zkj/Documents/Windows_Share/Windows_share_27F4/Board_workspace/rknn_board_test/rk_test_project/examples/rknn_yolo_demo/../libs/opencv/opencv410_aarch64/lib64/libopencv_photo.so.4.1"
    "/home/zkj/Documents/Windows_Share/Windows_share_27F4/Board_workspace/rknn_board_test/rk_test_project/examples/rknn_yolo_demo/../libs/opencv/opencv410_aarch64/lib64/libopencv_photo.so.4.1.0"
    "/home/zkj/Documents/Windows_Share/Windows_share_27F4/Board_workspace/rknn_board_test/rk_test_project/examples/rknn_yolo_demo/../libs/opencv/opencv410_aarch64/lib64/libopencv_stitching.so"
    "/home/zkj/Documents/Windows_Share/Windows_share_27F4/Board_workspace/rknn_board_test/rk_test_project/examples/rknn_yolo_demo/../libs/opencv/opencv410_aarch64/lib64/libopencv_stitching.so.4.1"
    "/home/zkj/Documents/Windows_Share/Windows_share_27F4/Board_workspace/rknn_board_test/rk_test_project/examples/rknn_yolo_demo/../libs/opencv/opencv410_aarch64/lib64/libopencv_stitching.so.4.1.0"
    "/home/zkj/Documents/Windows_Share/Windows_share_27F4/Board_workspace/rknn_board_test/rk_test_project/examples/rknn_yolo_demo/../libs/opencv/opencv410_aarch64/lib64/libopencv_video.so"
    "/home/zkj/Documents/Windows_Share/Windows_share_27F4/Board_workspace/rknn_board_test/rk_test_project/examples/rknn_yolo_demo/../libs/opencv/opencv410_aarch64/lib64/libopencv_video.so.4.1"
    "/home/zkj/Documents/Windows_Share/Windows_share_27F4/Board_workspace/rknn_board_test/rk_test_project/examples/rknn_yolo_demo/../libs/opencv/opencv410_aarch64/lib64/libopencv_video.so.4.1.0"
    "/home/zkj/Documents/Windows_Share/Windows_share_27F4/Board_workspace/rknn_board_test/rk_test_project/examples/rknn_yolo_demo/../libs/opencv/opencv410_aarch64/lib64/libopencv_videoio.so"
    "/home/zkj/Documents/Windows_Share/Windows_share_27F4/Board_workspace/rknn_board_test/rk_test_project/examples/rknn_yolo_demo/../libs/opencv/opencv410_aarch64/lib64/libopencv_videoio.so.4.1"
    "/home/zkj/Documents/Windows_Share/Windows_share_27F4/Board_workspace/rknn_board_test/rk_test_project/examples/rknn_yolo_demo/../libs/opencv/opencv410_aarch64/lib64/libopencv_videoio.so.4.1.0"
    "/home/zkj/Documents/Windows_Share/Windows_share_27F4/Board_workspace/rknn_board_test/rk_test_project/examples/rknn_yolo_demo/../libs/opencv/opencv410_aarch64/lib64/libz.so"
    "/home/zkj/Documents/Windows_Share/Windows_share_27F4/Board_workspace/rknn_board_test/rk_test_project/examples/rknn_yolo_demo/../libs/opencv/opencv410_aarch64/lib64/libz.so.1"
    "/home/zkj/Documents/Windows_Share/Windows_share_27F4/Board_workspace/rknn_board_test/rk_test_project/examples/rknn_yolo_demo/../libs/opencv/opencv410_aarch64/lib64/libz.so.1.2.11"
    )
endif()

if(CMAKE_INSTALL_COMPONENT)
  set(CMAKE_INSTALL_MANIFEST "install_manifest_${CMAKE_INSTALL_COMPONENT}.txt")
else()
  set(CMAKE_INSTALL_MANIFEST "install_manifest.txt")
endif()

string(REPLACE ";" "\n" CMAKE_INSTALL_MANIFEST_CONTENT
       "${CMAKE_INSTALL_MANIFEST_FILES}")
file(WRITE "/home/zkj/Documents/Windows_Share/Windows_share_27F4/Board_workspace/rknn_board_test/rk_test_project/examples/rknn_yolo_demo/build/${CMAKE_INSTALL_MANIFEST}"
     "${CMAKE_INSTALL_MANIFEST_CONTENT}")
