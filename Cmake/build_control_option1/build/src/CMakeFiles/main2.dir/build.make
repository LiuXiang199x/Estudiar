# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.10

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
CMAKE_SOURCE_DIR = /home/xiang/Estudiar/Cmake/build_control_option

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/xiang/Estudiar/Cmake/build_control_option/build

# Include any dependencies generated for this target.
include src/CMakeFiles/main2.dir/depend.make

# Include the progress variables for this target.
include src/CMakeFiles/main2.dir/progress.make

# Include the compile flags for this target's objects.
include src/CMakeFiles/main2.dir/flags.make

src/CMakeFiles/main2.dir/main2.cc.o: src/CMakeFiles/main2.dir/flags.make
src/CMakeFiles/main2.dir/main2.cc.o: ../src/main2.cc
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/xiang/Estudiar/Cmake/build_control_option/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object src/CMakeFiles/main2.dir/main2.cc.o"
	cd /home/xiang/Estudiar/Cmake/build_control_option/build/src && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/main2.dir/main2.cc.o -c /home/xiang/Estudiar/Cmake/build_control_option/src/main2.cc

src/CMakeFiles/main2.dir/main2.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/main2.dir/main2.cc.i"
	cd /home/xiang/Estudiar/Cmake/build_control_option/build/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/xiang/Estudiar/Cmake/build_control_option/src/main2.cc > CMakeFiles/main2.dir/main2.cc.i

src/CMakeFiles/main2.dir/main2.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/main2.dir/main2.cc.s"
	cd /home/xiang/Estudiar/Cmake/build_control_option/build/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/xiang/Estudiar/Cmake/build_control_option/src/main2.cc -o CMakeFiles/main2.dir/main2.cc.s

src/CMakeFiles/main2.dir/main2.cc.o.requires:

.PHONY : src/CMakeFiles/main2.dir/main2.cc.o.requires

src/CMakeFiles/main2.dir/main2.cc.o.provides: src/CMakeFiles/main2.dir/main2.cc.o.requires
	$(MAKE) -f src/CMakeFiles/main2.dir/build.make src/CMakeFiles/main2.dir/main2.cc.o.provides.build
.PHONY : src/CMakeFiles/main2.dir/main2.cc.o.provides

src/CMakeFiles/main2.dir/main2.cc.o.provides.build: src/CMakeFiles/main2.dir/main2.cc.o


# Object files for target main2
main2_OBJECTS = \
"CMakeFiles/main2.dir/main2.cc.o"

# External object files for target main2
main2_EXTERNAL_OBJECTS =

../bin/main2: src/CMakeFiles/main2.dir/main2.cc.o
../bin/main2: src/CMakeFiles/main2.dir/build.make
../bin/main2: src/CMakeFiles/main2.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/xiang/Estudiar/Cmake/build_control_option/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable ../../bin/main2"
	cd /home/xiang/Estudiar/Cmake/build_control_option/build/src && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/main2.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
src/CMakeFiles/main2.dir/build: ../bin/main2

.PHONY : src/CMakeFiles/main2.dir/build

src/CMakeFiles/main2.dir/requires: src/CMakeFiles/main2.dir/main2.cc.o.requires

.PHONY : src/CMakeFiles/main2.dir/requires

src/CMakeFiles/main2.dir/clean:
	cd /home/xiang/Estudiar/Cmake/build_control_option/build/src && $(CMAKE_COMMAND) -P CMakeFiles/main2.dir/cmake_clean.cmake
.PHONY : src/CMakeFiles/main2.dir/clean

src/CMakeFiles/main2.dir/depend:
	cd /home/xiang/Estudiar/Cmake/build_control_option/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/xiang/Estudiar/Cmake/build_control_option /home/xiang/Estudiar/Cmake/build_control_option/src /home/xiang/Estudiar/Cmake/build_control_option/build /home/xiang/Estudiar/Cmake/build_control_option/build/src /home/xiang/Estudiar/Cmake/build_control_option/build/src/CMakeFiles/main2.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : src/CMakeFiles/main2.dir/depend
