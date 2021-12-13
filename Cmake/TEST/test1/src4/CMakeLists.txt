aux_source_directory (. SRC_LIST)

include_directories(../inlude)

# add_executable (main ${SRC_LIST})
set (SRC_LIST ${PROJECT_SOURCE_DIR}/src4/main.cc)

add_library (1shared SHARED ${SRC_LIST})
add_library (1static STATIC ${SRC_LIST})

set (EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/src4)


