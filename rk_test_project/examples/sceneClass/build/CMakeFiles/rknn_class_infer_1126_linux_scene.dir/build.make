# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.16

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/agent/rk_test_project/examples/sceneClass

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/agent/rk_test_project/examples/sceneClass/build

# Include any dependencies generated for this target.
include CMakeFiles/rknn_class_infer_1126_linux_scene.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/rknn_class_infer_1126_linux_scene.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/rknn_class_infer_1126_linux_scene.dir/flags.make

CMakeFiles/rknn_class_infer_1126_linux_scene.dir/src/main.cc.o: CMakeFiles/rknn_class_infer_1126_linux_scene.dir/flags.make
CMakeFiles/rknn_class_infer_1126_linux_scene.dir/src/main.cc.o: ../src/main.cc
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/agent/rk_test_project/examples/sceneClass/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/rknn_class_infer_1126_linux_scene.dir/src/main.cc.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/rknn_class_infer_1126_linux_scene.dir/src/main.cc.o -c /home/agent/rk_test_project/examples/sceneClass/src/main.cc

CMakeFiles/rknn_class_infer_1126_linux_scene.dir/src/main.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/rknn_class_infer_1126_linux_scene.dir/src/main.cc.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/agent/rk_test_project/examples/sceneClass/src/main.cc > CMakeFiles/rknn_class_infer_1126_linux_scene.dir/src/main.cc.i

CMakeFiles/rknn_class_infer_1126_linux_scene.dir/src/main.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/rknn_class_infer_1126_linux_scene.dir/src/main.cc.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/agent/rk_test_project/examples/sceneClass/src/main.cc -o CMakeFiles/rknn_class_infer_1126_linux_scene.dir/src/main.cc.s

CMakeFiles/rknn_class_infer_1126_linux_scene.dir/src/SceneNet.cpp.o: CMakeFiles/rknn_class_infer_1126_linux_scene.dir/flags.make
CMakeFiles/rknn_class_infer_1126_linux_scene.dir/src/SceneNet.cpp.o: ../src/SceneNet.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/agent/rk_test_project/examples/sceneClass/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/rknn_class_infer_1126_linux_scene.dir/src/SceneNet.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/rknn_class_infer_1126_linux_scene.dir/src/SceneNet.cpp.o -c /home/agent/rk_test_project/examples/sceneClass/src/SceneNet.cpp

CMakeFiles/rknn_class_infer_1126_linux_scene.dir/src/SceneNet.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/rknn_class_infer_1126_linux_scene.dir/src/SceneNet.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/agent/rk_test_project/examples/sceneClass/src/SceneNet.cpp > CMakeFiles/rknn_class_infer_1126_linux_scene.dir/src/SceneNet.cpp.i

CMakeFiles/rknn_class_infer_1126_linux_scene.dir/src/SceneNet.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/rknn_class_infer_1126_linux_scene.dir/src/SceneNet.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/agent/rk_test_project/examples/sceneClass/src/SceneNet.cpp -o CMakeFiles/rknn_class_infer_1126_linux_scene.dir/src/SceneNet.cpp.s

# Object files for target rknn_class_infer_1126_linux_scene
rknn_class_infer_1126_linux_scene_OBJECTS = \
"CMakeFiles/rknn_class_infer_1126_linux_scene.dir/src/main.cc.o" \
"CMakeFiles/rknn_class_infer_1126_linux_scene.dir/src/SceneNet.cpp.o"

# External object files for target rknn_class_infer_1126_linux_scene
rknn_class_infer_1126_linux_scene_EXTERNAL_OBJECTS =

rknn_class_infer_1126_linux_scene: CMakeFiles/rknn_class_infer_1126_linux_scene.dir/src/main.cc.o
rknn_class_infer_1126_linux_scene: CMakeFiles/rknn_class_infer_1126_linux_scene.dir/src/SceneNet.cpp.o
rknn_class_infer_1126_linux_scene: CMakeFiles/rknn_class_infer_1126_linux_scene.dir/build.make
rknn_class_infer_1126_linux_scene: /home/agent/rk_test_project/librknn_api/lib64/librknn_api.so
rknn_class_infer_1126_linux_scene: /home/agent/rk_test_project/examples/libs/opencv/opencv410_aarch64/lib64/libopencv_calib3d.so
rknn_class_infer_1126_linux_scene: /home/agent/rk_test_project/examples/libs/opencv/opencv410_aarch64/lib64/libopencv_calib3d.so.4.1
rknn_class_infer_1126_linux_scene: /home/agent/rk_test_project/examples/libs/opencv/opencv410_aarch64/lib64/libopencv_calib3d.so.4.1.0
rknn_class_infer_1126_linux_scene: /home/agent/rk_test_project/examples/libs/opencv/opencv410_aarch64/lib64/libopencv_core.so
rknn_class_infer_1126_linux_scene: /home/agent/rk_test_project/examples/libs/opencv/opencv410_aarch64/lib64/libopencv_core.so.4.1
rknn_class_infer_1126_linux_scene: /home/agent/rk_test_project/examples/libs/opencv/opencv410_aarch64/lib64/libopencv_core.so.4.1.0
rknn_class_infer_1126_linux_scene: /home/agent/rk_test_project/examples/libs/opencv/opencv410_aarch64/lib64/libopencv_dnn.so
rknn_class_infer_1126_linux_scene: /home/agent/rk_test_project/examples/libs/opencv/opencv410_aarch64/lib64/libopencv_dnn.so.4.1
rknn_class_infer_1126_linux_scene: /home/agent/rk_test_project/examples/libs/opencv/opencv410_aarch64/lib64/libopencv_dnn.so.4.1.0
rknn_class_infer_1126_linux_scene: /home/agent/rk_test_project/examples/libs/opencv/opencv410_aarch64/lib64/libopencv_features2d.so
rknn_class_infer_1126_linux_scene: /home/agent/rk_test_project/examples/libs/opencv/opencv410_aarch64/lib64/libopencv_features2d.so.4.1
rknn_class_infer_1126_linux_scene: /home/agent/rk_test_project/examples/libs/opencv/opencv410_aarch64/lib64/libopencv_features2d.so.4.1.0
rknn_class_infer_1126_linux_scene: /home/agent/rk_test_project/examples/libs/opencv/opencv410_aarch64/lib64/libopencv_flann.so
rknn_class_infer_1126_linux_scene: /home/agent/rk_test_project/examples/libs/opencv/opencv410_aarch64/lib64/libopencv_flann.so.4.1
rknn_class_infer_1126_linux_scene: /home/agent/rk_test_project/examples/libs/opencv/opencv410_aarch64/lib64/libopencv_flann.so.4.1.0
rknn_class_infer_1126_linux_scene: /home/agent/rk_test_project/examples/libs/opencv/opencv410_aarch64/lib64/libopencv_gapi.so
rknn_class_infer_1126_linux_scene: /home/agent/rk_test_project/examples/libs/opencv/opencv410_aarch64/lib64/libopencv_gapi.so.4.1
rknn_class_infer_1126_linux_scene: /home/agent/rk_test_project/examples/libs/opencv/opencv410_aarch64/lib64/libopencv_gapi.so.4.1.0
rknn_class_infer_1126_linux_scene: /home/agent/rk_test_project/examples/libs/opencv/opencv410_aarch64/lib64/libopencv_highgui.so
rknn_class_infer_1126_linux_scene: /home/agent/rk_test_project/examples/libs/opencv/opencv410_aarch64/lib64/libopencv_highgui.so.4.1
rknn_class_infer_1126_linux_scene: /home/agent/rk_test_project/examples/libs/opencv/opencv410_aarch64/lib64/libopencv_highgui.so.4.1.0
rknn_class_infer_1126_linux_scene: /home/agent/rk_test_project/examples/libs/opencv/opencv410_aarch64/lib64/libopencv_imgcodecs.so
rknn_class_infer_1126_linux_scene: /home/agent/rk_test_project/examples/libs/opencv/opencv410_aarch64/lib64/libopencv_imgcodecs.so.4.1
rknn_class_infer_1126_linux_scene: /home/agent/rk_test_project/examples/libs/opencv/opencv410_aarch64/lib64/libopencv_imgcodecs.so.4.1.0
rknn_class_infer_1126_linux_scene: /home/agent/rk_test_project/examples/libs/opencv/opencv410_aarch64/lib64/libopencv_imgproc.so
rknn_class_infer_1126_linux_scene: /home/agent/rk_test_project/examples/libs/opencv/opencv410_aarch64/lib64/libopencv_imgproc.so.4.1
rknn_class_infer_1126_linux_scene: /home/agent/rk_test_project/examples/libs/opencv/opencv410_aarch64/lib64/libopencv_imgproc.so.4.1.0
rknn_class_infer_1126_linux_scene: /home/agent/rk_test_project/examples/libs/opencv/opencv410_aarch64/lib64/libopencv_ml.so
rknn_class_infer_1126_linux_scene: /home/agent/rk_test_project/examples/libs/opencv/opencv410_aarch64/lib64/libopencv_ml.so.4.1
rknn_class_infer_1126_linux_scene: /home/agent/rk_test_project/examples/libs/opencv/opencv410_aarch64/lib64/libopencv_ml.so.4.1.0
rknn_class_infer_1126_linux_scene: /home/agent/rk_test_project/examples/libs/opencv/opencv410_aarch64/lib64/libopencv_objdetect.so
rknn_class_infer_1126_linux_scene: /home/agent/rk_test_project/examples/libs/opencv/opencv410_aarch64/lib64/libopencv_objdetect.so.4.1
rknn_class_infer_1126_linux_scene: /home/agent/rk_test_project/examples/libs/opencv/opencv410_aarch64/lib64/libopencv_objdetect.so.4.1.0
rknn_class_infer_1126_linux_scene: /home/agent/rk_test_project/examples/libs/opencv/opencv410_aarch64/lib64/libopencv_photo.so
rknn_class_infer_1126_linux_scene: /home/agent/rk_test_project/examples/libs/opencv/opencv410_aarch64/lib64/libopencv_photo.so.4.1
rknn_class_infer_1126_linux_scene: /home/agent/rk_test_project/examples/libs/opencv/opencv410_aarch64/lib64/libopencv_photo.so.4.1.0
rknn_class_infer_1126_linux_scene: /home/agent/rk_test_project/examples/libs/opencv/opencv410_aarch64/lib64/libopencv_stitching.so
rknn_class_infer_1126_linux_scene: /home/agent/rk_test_project/examples/libs/opencv/opencv410_aarch64/lib64/libopencv_stitching.so.4.1
rknn_class_infer_1126_linux_scene: /home/agent/rk_test_project/examples/libs/opencv/opencv410_aarch64/lib64/libopencv_stitching.so.4.1.0
rknn_class_infer_1126_linux_scene: /home/agent/rk_test_project/examples/libs/opencv/opencv410_aarch64/lib64/libopencv_video.so
rknn_class_infer_1126_linux_scene: /home/agent/rk_test_project/examples/libs/opencv/opencv410_aarch64/lib64/libopencv_video.so.4.1
rknn_class_infer_1126_linux_scene: /home/agent/rk_test_project/examples/libs/opencv/opencv410_aarch64/lib64/libopencv_video.so.4.1.0
rknn_class_infer_1126_linux_scene: /home/agent/rk_test_project/examples/libs/opencv/opencv410_aarch64/lib64/libopencv_videoio.so
rknn_class_infer_1126_linux_scene: /home/agent/rk_test_project/examples/libs/opencv/opencv410_aarch64/lib64/libopencv_videoio.so.4.1
rknn_class_infer_1126_linux_scene: /home/agent/rk_test_project/examples/libs/opencv/opencv410_aarch64/lib64/libopencv_videoio.so.4.1.0
rknn_class_infer_1126_linux_scene: /home/agent/rk_test_project/examples/libs/opencv/opencv410_aarch64/lib64/libz.so
rknn_class_infer_1126_linux_scene: /home/agent/rk_test_project/examples/libs/opencv/opencv410_aarch64/lib64/libz.so.1
rknn_class_infer_1126_linux_scene: /home/agent/rk_test_project/examples/libs/opencv/opencv410_aarch64/lib64/libz.so.1.2.11
rknn_class_infer_1126_linux_scene: CMakeFiles/rknn_class_infer_1126_linux_scene.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/agent/rk_test_project/examples/sceneClass/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Linking CXX executable rknn_class_infer_1126_linux_scene"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/rknn_class_infer_1126_linux_scene.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/rknn_class_infer_1126_linux_scene.dir/build: rknn_class_infer_1126_linux_scene

.PHONY : CMakeFiles/rknn_class_infer_1126_linux_scene.dir/build

CMakeFiles/rknn_class_infer_1126_linux_scene.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/rknn_class_infer_1126_linux_scene.dir/cmake_clean.cmake
.PHONY : CMakeFiles/rknn_class_infer_1126_linux_scene.dir/clean

CMakeFiles/rknn_class_infer_1126_linux_scene.dir/depend:
	cd /home/agent/rk_test_project/examples/sceneClass/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/agent/rk_test_project/examples/sceneClass /home/agent/rk_test_project/examples/sceneClass /home/agent/rk_test_project/examples/sceneClass/build /home/agent/rk_test_project/examples/sceneClass/build /home/agent/rk_test_project/examples/sceneClass/build/CMakeFiles/rknn_class_infer_1126_linux_scene.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/rknn_class_infer_1126_linux_scene.dir/depend

