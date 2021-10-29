# generated from genmsg/cmake/pkg-genmsg.cmake.em

message(STATUS "ros_tutorial: 7 messages, 1 services")

set(MSG_I_FLAGS "-Iros_tutorial:/home/xiang/test_ros/devel/share/ros_tutorial/msg;-Istd_msgs:/opt/ros/melodic/share/std_msgs/cmake/../msg;-Iactionlib_msgs:/opt/ros/melodic/share/actionlib_msgs/cmake/../msg")

# Find all generators
find_package(gencpp REQUIRED)
find_package(geneus REQUIRED)
find_package(genlisp REQUIRED)
find_package(gennodejs REQUIRED)
find_package(genpy REQUIRED)

add_custom_target(ros_tutorial_generate_messages ALL)

# verify that message/service dependencies have not changed since configure



get_filename_component(_filename "/home/xiang/test_ros/devel/share/ros_tutorial/msg/counterAction.msg" NAME_WE)
add_custom_target(_ros_tutorial_generate_messages_check_deps_${_filename}
  COMMAND ${CATKIN_ENV} ${PYTHON_EXECUTABLE} ${GENMSG_CHECK_DEPS_SCRIPT} "ros_tutorial" "/home/xiang/test_ros/devel/share/ros_tutorial/msg/counterAction.msg" "actionlib_msgs/GoalID:ros_tutorial/counterActionGoal:actionlib_msgs/GoalStatus:ros_tutorial/counterResult:ros_tutorial/counterFeedback:ros_tutorial/counterActionResult:std_msgs/Header:ros_tutorial/counterActionFeedback:ros_tutorial/counterGoal"
)

get_filename_component(_filename "/home/xiang/test_ros/devel/share/ros_tutorial/msg/counterResult.msg" NAME_WE)
add_custom_target(_ros_tutorial_generate_messages_check_deps_${_filename}
  COMMAND ${CATKIN_ENV} ${PYTHON_EXECUTABLE} ${GENMSG_CHECK_DEPS_SCRIPT} "ros_tutorial" "/home/xiang/test_ros/devel/share/ros_tutorial/msg/counterResult.msg" ""
)

get_filename_component(_filename "/home/xiang/test_ros/devel/share/ros_tutorial/msg/counterActionFeedback.msg" NAME_WE)
add_custom_target(_ros_tutorial_generate_messages_check_deps_${_filename}
  COMMAND ${CATKIN_ENV} ${PYTHON_EXECUTABLE} ${GENMSG_CHECK_DEPS_SCRIPT} "ros_tutorial" "/home/xiang/test_ros/devel/share/ros_tutorial/msg/counterActionFeedback.msg" "ros_tutorial/counterFeedback:actionlib_msgs/GoalID:actionlib_msgs/GoalStatus:std_msgs/Header"
)

get_filename_component(_filename "/home/xiang/test_ros/devel/share/ros_tutorial/msg/counterFeedback.msg" NAME_WE)
add_custom_target(_ros_tutorial_generate_messages_check_deps_${_filename}
  COMMAND ${CATKIN_ENV} ${PYTHON_EXECUTABLE} ${GENMSG_CHECK_DEPS_SCRIPT} "ros_tutorial" "/home/xiang/test_ros/devel/share/ros_tutorial/msg/counterFeedback.msg" ""
)

get_filename_component(_filename "/home/xiang/test_ros/devel/share/ros_tutorial/msg/counterActionGoal.msg" NAME_WE)
add_custom_target(_ros_tutorial_generate_messages_check_deps_${_filename}
  COMMAND ${CATKIN_ENV} ${PYTHON_EXECUTABLE} ${GENMSG_CHECK_DEPS_SCRIPT} "ros_tutorial" "/home/xiang/test_ros/devel/share/ros_tutorial/msg/counterActionGoal.msg" "actionlib_msgs/GoalID:ros_tutorial/counterGoal:std_msgs/Header"
)

get_filename_component(_filename "/home/xiang/test_ros/src/ros_tutorial/srv/add.srv" NAME_WE)
add_custom_target(_ros_tutorial_generate_messages_check_deps_${_filename}
  COMMAND ${CATKIN_ENV} ${PYTHON_EXECUTABLE} ${GENMSG_CHECK_DEPS_SCRIPT} "ros_tutorial" "/home/xiang/test_ros/src/ros_tutorial/srv/add.srv" ""
)

get_filename_component(_filename "/home/xiang/test_ros/devel/share/ros_tutorial/msg/counterGoal.msg" NAME_WE)
add_custom_target(_ros_tutorial_generate_messages_check_deps_${_filename}
  COMMAND ${CATKIN_ENV} ${PYTHON_EXECUTABLE} ${GENMSG_CHECK_DEPS_SCRIPT} "ros_tutorial" "/home/xiang/test_ros/devel/share/ros_tutorial/msg/counterGoal.msg" ""
)

get_filename_component(_filename "/home/xiang/test_ros/devel/share/ros_tutorial/msg/counterActionResult.msg" NAME_WE)
add_custom_target(_ros_tutorial_generate_messages_check_deps_${_filename}
  COMMAND ${CATKIN_ENV} ${PYTHON_EXECUTABLE} ${GENMSG_CHECK_DEPS_SCRIPT} "ros_tutorial" "/home/xiang/test_ros/devel/share/ros_tutorial/msg/counterActionResult.msg" "ros_tutorial/counterResult:actionlib_msgs/GoalID:actionlib_msgs/GoalStatus:std_msgs/Header"
)

#
#  langs = gencpp;geneus;genlisp;gennodejs;genpy
#

### Section generating for lang: gencpp
### Generating Messages
_generate_msg_cpp(ros_tutorial
  "/home/xiang/test_ros/devel/share/ros_tutorial/msg/counterAction.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/melodic/share/actionlib_msgs/cmake/../msg/GoalID.msg;/home/xiang/test_ros/devel/share/ros_tutorial/msg/counterActionGoal.msg;/opt/ros/melodic/share/actionlib_msgs/cmake/../msg/GoalStatus.msg;/home/xiang/test_ros/devel/share/ros_tutorial/msg/counterResult.msg;/home/xiang/test_ros/devel/share/ros_tutorial/msg/counterFeedback.msg;/home/xiang/test_ros/devel/share/ros_tutorial/msg/counterActionResult.msg;/opt/ros/melodic/share/std_msgs/cmake/../msg/Header.msg;/home/xiang/test_ros/devel/share/ros_tutorial/msg/counterActionFeedback.msg;/home/xiang/test_ros/devel/share/ros_tutorial/msg/counterGoal.msg"
  ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/ros_tutorial
)
_generate_msg_cpp(ros_tutorial
  "/home/xiang/test_ros/devel/share/ros_tutorial/msg/counterResult.msg"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/ros_tutorial
)
_generate_msg_cpp(ros_tutorial
  "/home/xiang/test_ros/devel/share/ros_tutorial/msg/counterActionFeedback.msg"
  "${MSG_I_FLAGS}"
  "/home/xiang/test_ros/devel/share/ros_tutorial/msg/counterFeedback.msg;/opt/ros/melodic/share/actionlib_msgs/cmake/../msg/GoalID.msg;/opt/ros/melodic/share/actionlib_msgs/cmake/../msg/GoalStatus.msg;/opt/ros/melodic/share/std_msgs/cmake/../msg/Header.msg"
  ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/ros_tutorial
)
_generate_msg_cpp(ros_tutorial
  "/home/xiang/test_ros/devel/share/ros_tutorial/msg/counterFeedback.msg"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/ros_tutorial
)
_generate_msg_cpp(ros_tutorial
  "/home/xiang/test_ros/devel/share/ros_tutorial/msg/counterActionGoal.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/melodic/share/actionlib_msgs/cmake/../msg/GoalID.msg;/home/xiang/test_ros/devel/share/ros_tutorial/msg/counterGoal.msg;/opt/ros/melodic/share/std_msgs/cmake/../msg/Header.msg"
  ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/ros_tutorial
)
_generate_msg_cpp(ros_tutorial
  "/home/xiang/test_ros/devel/share/ros_tutorial/msg/counterGoal.msg"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/ros_tutorial
)
_generate_msg_cpp(ros_tutorial
  "/home/xiang/test_ros/devel/share/ros_tutorial/msg/counterActionResult.msg"
  "${MSG_I_FLAGS}"
  "/home/xiang/test_ros/devel/share/ros_tutorial/msg/counterResult.msg;/opt/ros/melodic/share/actionlib_msgs/cmake/../msg/GoalID.msg;/opt/ros/melodic/share/actionlib_msgs/cmake/../msg/GoalStatus.msg;/opt/ros/melodic/share/std_msgs/cmake/../msg/Header.msg"
  ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/ros_tutorial
)

### Generating Services
_generate_srv_cpp(ros_tutorial
  "/home/xiang/test_ros/src/ros_tutorial/srv/add.srv"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/ros_tutorial
)

### Generating Module File
_generate_module_cpp(ros_tutorial
  ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/ros_tutorial
  "${ALL_GEN_OUTPUT_FILES_cpp}"
)

add_custom_target(ros_tutorial_generate_messages_cpp
  DEPENDS ${ALL_GEN_OUTPUT_FILES_cpp}
)
add_dependencies(ros_tutorial_generate_messages ros_tutorial_generate_messages_cpp)

# add dependencies to all check dependencies targets
get_filename_component(_filename "/home/xiang/test_ros/devel/share/ros_tutorial/msg/counterAction.msg" NAME_WE)
add_dependencies(ros_tutorial_generate_messages_cpp _ros_tutorial_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/xiang/test_ros/devel/share/ros_tutorial/msg/counterResult.msg" NAME_WE)
add_dependencies(ros_tutorial_generate_messages_cpp _ros_tutorial_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/xiang/test_ros/devel/share/ros_tutorial/msg/counterActionFeedback.msg" NAME_WE)
add_dependencies(ros_tutorial_generate_messages_cpp _ros_tutorial_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/xiang/test_ros/devel/share/ros_tutorial/msg/counterFeedback.msg" NAME_WE)
add_dependencies(ros_tutorial_generate_messages_cpp _ros_tutorial_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/xiang/test_ros/devel/share/ros_tutorial/msg/counterActionGoal.msg" NAME_WE)
add_dependencies(ros_tutorial_generate_messages_cpp _ros_tutorial_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/xiang/test_ros/src/ros_tutorial/srv/add.srv" NAME_WE)
add_dependencies(ros_tutorial_generate_messages_cpp _ros_tutorial_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/xiang/test_ros/devel/share/ros_tutorial/msg/counterGoal.msg" NAME_WE)
add_dependencies(ros_tutorial_generate_messages_cpp _ros_tutorial_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/xiang/test_ros/devel/share/ros_tutorial/msg/counterActionResult.msg" NAME_WE)
add_dependencies(ros_tutorial_generate_messages_cpp _ros_tutorial_generate_messages_check_deps_${_filename})

# target for backward compatibility
add_custom_target(ros_tutorial_gencpp)
add_dependencies(ros_tutorial_gencpp ros_tutorial_generate_messages_cpp)

# register target for catkin_package(EXPORTED_TARGETS)
list(APPEND ${PROJECT_NAME}_EXPORTED_TARGETS ros_tutorial_generate_messages_cpp)

### Section generating for lang: geneus
### Generating Messages
_generate_msg_eus(ros_tutorial
  "/home/xiang/test_ros/devel/share/ros_tutorial/msg/counterAction.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/melodic/share/actionlib_msgs/cmake/../msg/GoalID.msg;/home/xiang/test_ros/devel/share/ros_tutorial/msg/counterActionGoal.msg;/opt/ros/melodic/share/actionlib_msgs/cmake/../msg/GoalStatus.msg;/home/xiang/test_ros/devel/share/ros_tutorial/msg/counterResult.msg;/home/xiang/test_ros/devel/share/ros_tutorial/msg/counterFeedback.msg;/home/xiang/test_ros/devel/share/ros_tutorial/msg/counterActionResult.msg;/opt/ros/melodic/share/std_msgs/cmake/../msg/Header.msg;/home/xiang/test_ros/devel/share/ros_tutorial/msg/counterActionFeedback.msg;/home/xiang/test_ros/devel/share/ros_tutorial/msg/counterGoal.msg"
  ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/ros_tutorial
)
_generate_msg_eus(ros_tutorial
  "/home/xiang/test_ros/devel/share/ros_tutorial/msg/counterResult.msg"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/ros_tutorial
)
_generate_msg_eus(ros_tutorial
  "/home/xiang/test_ros/devel/share/ros_tutorial/msg/counterActionFeedback.msg"
  "${MSG_I_FLAGS}"
  "/home/xiang/test_ros/devel/share/ros_tutorial/msg/counterFeedback.msg;/opt/ros/melodic/share/actionlib_msgs/cmake/../msg/GoalID.msg;/opt/ros/melodic/share/actionlib_msgs/cmake/../msg/GoalStatus.msg;/opt/ros/melodic/share/std_msgs/cmake/../msg/Header.msg"
  ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/ros_tutorial
)
_generate_msg_eus(ros_tutorial
  "/home/xiang/test_ros/devel/share/ros_tutorial/msg/counterFeedback.msg"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/ros_tutorial
)
_generate_msg_eus(ros_tutorial
  "/home/xiang/test_ros/devel/share/ros_tutorial/msg/counterActionGoal.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/melodic/share/actionlib_msgs/cmake/../msg/GoalID.msg;/home/xiang/test_ros/devel/share/ros_tutorial/msg/counterGoal.msg;/opt/ros/melodic/share/std_msgs/cmake/../msg/Header.msg"
  ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/ros_tutorial
)
_generate_msg_eus(ros_tutorial
  "/home/xiang/test_ros/devel/share/ros_tutorial/msg/counterGoal.msg"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/ros_tutorial
)
_generate_msg_eus(ros_tutorial
  "/home/xiang/test_ros/devel/share/ros_tutorial/msg/counterActionResult.msg"
  "${MSG_I_FLAGS}"
  "/home/xiang/test_ros/devel/share/ros_tutorial/msg/counterResult.msg;/opt/ros/melodic/share/actionlib_msgs/cmake/../msg/GoalID.msg;/opt/ros/melodic/share/actionlib_msgs/cmake/../msg/GoalStatus.msg;/opt/ros/melodic/share/std_msgs/cmake/../msg/Header.msg"
  ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/ros_tutorial
)

### Generating Services
_generate_srv_eus(ros_tutorial
  "/home/xiang/test_ros/src/ros_tutorial/srv/add.srv"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/ros_tutorial
)

### Generating Module File
_generate_module_eus(ros_tutorial
  ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/ros_tutorial
  "${ALL_GEN_OUTPUT_FILES_eus}"
)

add_custom_target(ros_tutorial_generate_messages_eus
  DEPENDS ${ALL_GEN_OUTPUT_FILES_eus}
)
add_dependencies(ros_tutorial_generate_messages ros_tutorial_generate_messages_eus)

# add dependencies to all check dependencies targets
get_filename_component(_filename "/home/xiang/test_ros/devel/share/ros_tutorial/msg/counterAction.msg" NAME_WE)
add_dependencies(ros_tutorial_generate_messages_eus _ros_tutorial_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/xiang/test_ros/devel/share/ros_tutorial/msg/counterResult.msg" NAME_WE)
add_dependencies(ros_tutorial_generate_messages_eus _ros_tutorial_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/xiang/test_ros/devel/share/ros_tutorial/msg/counterActionFeedback.msg" NAME_WE)
add_dependencies(ros_tutorial_generate_messages_eus _ros_tutorial_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/xiang/test_ros/devel/share/ros_tutorial/msg/counterFeedback.msg" NAME_WE)
add_dependencies(ros_tutorial_generate_messages_eus _ros_tutorial_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/xiang/test_ros/devel/share/ros_tutorial/msg/counterActionGoal.msg" NAME_WE)
add_dependencies(ros_tutorial_generate_messages_eus _ros_tutorial_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/xiang/test_ros/src/ros_tutorial/srv/add.srv" NAME_WE)
add_dependencies(ros_tutorial_generate_messages_eus _ros_tutorial_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/xiang/test_ros/devel/share/ros_tutorial/msg/counterGoal.msg" NAME_WE)
add_dependencies(ros_tutorial_generate_messages_eus _ros_tutorial_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/xiang/test_ros/devel/share/ros_tutorial/msg/counterActionResult.msg" NAME_WE)
add_dependencies(ros_tutorial_generate_messages_eus _ros_tutorial_generate_messages_check_deps_${_filename})

# target for backward compatibility
add_custom_target(ros_tutorial_geneus)
add_dependencies(ros_tutorial_geneus ros_tutorial_generate_messages_eus)

# register target for catkin_package(EXPORTED_TARGETS)
list(APPEND ${PROJECT_NAME}_EXPORTED_TARGETS ros_tutorial_generate_messages_eus)

### Section generating for lang: genlisp
### Generating Messages
_generate_msg_lisp(ros_tutorial
  "/home/xiang/test_ros/devel/share/ros_tutorial/msg/counterAction.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/melodic/share/actionlib_msgs/cmake/../msg/GoalID.msg;/home/xiang/test_ros/devel/share/ros_tutorial/msg/counterActionGoal.msg;/opt/ros/melodic/share/actionlib_msgs/cmake/../msg/GoalStatus.msg;/home/xiang/test_ros/devel/share/ros_tutorial/msg/counterResult.msg;/home/xiang/test_ros/devel/share/ros_tutorial/msg/counterFeedback.msg;/home/xiang/test_ros/devel/share/ros_tutorial/msg/counterActionResult.msg;/opt/ros/melodic/share/std_msgs/cmake/../msg/Header.msg;/home/xiang/test_ros/devel/share/ros_tutorial/msg/counterActionFeedback.msg;/home/xiang/test_ros/devel/share/ros_tutorial/msg/counterGoal.msg"
  ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/ros_tutorial
)
_generate_msg_lisp(ros_tutorial
  "/home/xiang/test_ros/devel/share/ros_tutorial/msg/counterResult.msg"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/ros_tutorial
)
_generate_msg_lisp(ros_tutorial
  "/home/xiang/test_ros/devel/share/ros_tutorial/msg/counterActionFeedback.msg"
  "${MSG_I_FLAGS}"
  "/home/xiang/test_ros/devel/share/ros_tutorial/msg/counterFeedback.msg;/opt/ros/melodic/share/actionlib_msgs/cmake/../msg/GoalID.msg;/opt/ros/melodic/share/actionlib_msgs/cmake/../msg/GoalStatus.msg;/opt/ros/melodic/share/std_msgs/cmake/../msg/Header.msg"
  ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/ros_tutorial
)
_generate_msg_lisp(ros_tutorial
  "/home/xiang/test_ros/devel/share/ros_tutorial/msg/counterFeedback.msg"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/ros_tutorial
)
_generate_msg_lisp(ros_tutorial
  "/home/xiang/test_ros/devel/share/ros_tutorial/msg/counterActionGoal.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/melodic/share/actionlib_msgs/cmake/../msg/GoalID.msg;/home/xiang/test_ros/devel/share/ros_tutorial/msg/counterGoal.msg;/opt/ros/melodic/share/std_msgs/cmake/../msg/Header.msg"
  ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/ros_tutorial
)
_generate_msg_lisp(ros_tutorial
  "/home/xiang/test_ros/devel/share/ros_tutorial/msg/counterGoal.msg"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/ros_tutorial
)
_generate_msg_lisp(ros_tutorial
  "/home/xiang/test_ros/devel/share/ros_tutorial/msg/counterActionResult.msg"
  "${MSG_I_FLAGS}"
  "/home/xiang/test_ros/devel/share/ros_tutorial/msg/counterResult.msg;/opt/ros/melodic/share/actionlib_msgs/cmake/../msg/GoalID.msg;/opt/ros/melodic/share/actionlib_msgs/cmake/../msg/GoalStatus.msg;/opt/ros/melodic/share/std_msgs/cmake/../msg/Header.msg"
  ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/ros_tutorial
)

### Generating Services
_generate_srv_lisp(ros_tutorial
  "/home/xiang/test_ros/src/ros_tutorial/srv/add.srv"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/ros_tutorial
)

### Generating Module File
_generate_module_lisp(ros_tutorial
  ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/ros_tutorial
  "${ALL_GEN_OUTPUT_FILES_lisp}"
)

add_custom_target(ros_tutorial_generate_messages_lisp
  DEPENDS ${ALL_GEN_OUTPUT_FILES_lisp}
)
add_dependencies(ros_tutorial_generate_messages ros_tutorial_generate_messages_lisp)

# add dependencies to all check dependencies targets
get_filename_component(_filename "/home/xiang/test_ros/devel/share/ros_tutorial/msg/counterAction.msg" NAME_WE)
add_dependencies(ros_tutorial_generate_messages_lisp _ros_tutorial_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/xiang/test_ros/devel/share/ros_tutorial/msg/counterResult.msg" NAME_WE)
add_dependencies(ros_tutorial_generate_messages_lisp _ros_tutorial_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/xiang/test_ros/devel/share/ros_tutorial/msg/counterActionFeedback.msg" NAME_WE)
add_dependencies(ros_tutorial_generate_messages_lisp _ros_tutorial_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/xiang/test_ros/devel/share/ros_tutorial/msg/counterFeedback.msg" NAME_WE)
add_dependencies(ros_tutorial_generate_messages_lisp _ros_tutorial_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/xiang/test_ros/devel/share/ros_tutorial/msg/counterActionGoal.msg" NAME_WE)
add_dependencies(ros_tutorial_generate_messages_lisp _ros_tutorial_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/xiang/test_ros/src/ros_tutorial/srv/add.srv" NAME_WE)
add_dependencies(ros_tutorial_generate_messages_lisp _ros_tutorial_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/xiang/test_ros/devel/share/ros_tutorial/msg/counterGoal.msg" NAME_WE)
add_dependencies(ros_tutorial_generate_messages_lisp _ros_tutorial_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/xiang/test_ros/devel/share/ros_tutorial/msg/counterActionResult.msg" NAME_WE)
add_dependencies(ros_tutorial_generate_messages_lisp _ros_tutorial_generate_messages_check_deps_${_filename})

# target for backward compatibility
add_custom_target(ros_tutorial_genlisp)
add_dependencies(ros_tutorial_genlisp ros_tutorial_generate_messages_lisp)

# register target for catkin_package(EXPORTED_TARGETS)
list(APPEND ${PROJECT_NAME}_EXPORTED_TARGETS ros_tutorial_generate_messages_lisp)

### Section generating for lang: gennodejs
### Generating Messages
_generate_msg_nodejs(ros_tutorial
  "/home/xiang/test_ros/devel/share/ros_tutorial/msg/counterAction.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/melodic/share/actionlib_msgs/cmake/../msg/GoalID.msg;/home/xiang/test_ros/devel/share/ros_tutorial/msg/counterActionGoal.msg;/opt/ros/melodic/share/actionlib_msgs/cmake/../msg/GoalStatus.msg;/home/xiang/test_ros/devel/share/ros_tutorial/msg/counterResult.msg;/home/xiang/test_ros/devel/share/ros_tutorial/msg/counterFeedback.msg;/home/xiang/test_ros/devel/share/ros_tutorial/msg/counterActionResult.msg;/opt/ros/melodic/share/std_msgs/cmake/../msg/Header.msg;/home/xiang/test_ros/devel/share/ros_tutorial/msg/counterActionFeedback.msg;/home/xiang/test_ros/devel/share/ros_tutorial/msg/counterGoal.msg"
  ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/ros_tutorial
)
_generate_msg_nodejs(ros_tutorial
  "/home/xiang/test_ros/devel/share/ros_tutorial/msg/counterResult.msg"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/ros_tutorial
)
_generate_msg_nodejs(ros_tutorial
  "/home/xiang/test_ros/devel/share/ros_tutorial/msg/counterActionFeedback.msg"
  "${MSG_I_FLAGS}"
  "/home/xiang/test_ros/devel/share/ros_tutorial/msg/counterFeedback.msg;/opt/ros/melodic/share/actionlib_msgs/cmake/../msg/GoalID.msg;/opt/ros/melodic/share/actionlib_msgs/cmake/../msg/GoalStatus.msg;/opt/ros/melodic/share/std_msgs/cmake/../msg/Header.msg"
  ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/ros_tutorial
)
_generate_msg_nodejs(ros_tutorial
  "/home/xiang/test_ros/devel/share/ros_tutorial/msg/counterFeedback.msg"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/ros_tutorial
)
_generate_msg_nodejs(ros_tutorial
  "/home/xiang/test_ros/devel/share/ros_tutorial/msg/counterActionGoal.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/melodic/share/actionlib_msgs/cmake/../msg/GoalID.msg;/home/xiang/test_ros/devel/share/ros_tutorial/msg/counterGoal.msg;/opt/ros/melodic/share/std_msgs/cmake/../msg/Header.msg"
  ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/ros_tutorial
)
_generate_msg_nodejs(ros_tutorial
  "/home/xiang/test_ros/devel/share/ros_tutorial/msg/counterGoal.msg"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/ros_tutorial
)
_generate_msg_nodejs(ros_tutorial
  "/home/xiang/test_ros/devel/share/ros_tutorial/msg/counterActionResult.msg"
  "${MSG_I_FLAGS}"
  "/home/xiang/test_ros/devel/share/ros_tutorial/msg/counterResult.msg;/opt/ros/melodic/share/actionlib_msgs/cmake/../msg/GoalID.msg;/opt/ros/melodic/share/actionlib_msgs/cmake/../msg/GoalStatus.msg;/opt/ros/melodic/share/std_msgs/cmake/../msg/Header.msg"
  ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/ros_tutorial
)

### Generating Services
_generate_srv_nodejs(ros_tutorial
  "/home/xiang/test_ros/src/ros_tutorial/srv/add.srv"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/ros_tutorial
)

### Generating Module File
_generate_module_nodejs(ros_tutorial
  ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/ros_tutorial
  "${ALL_GEN_OUTPUT_FILES_nodejs}"
)

add_custom_target(ros_tutorial_generate_messages_nodejs
  DEPENDS ${ALL_GEN_OUTPUT_FILES_nodejs}
)
add_dependencies(ros_tutorial_generate_messages ros_tutorial_generate_messages_nodejs)

# add dependencies to all check dependencies targets
get_filename_component(_filename "/home/xiang/test_ros/devel/share/ros_tutorial/msg/counterAction.msg" NAME_WE)
add_dependencies(ros_tutorial_generate_messages_nodejs _ros_tutorial_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/xiang/test_ros/devel/share/ros_tutorial/msg/counterResult.msg" NAME_WE)
add_dependencies(ros_tutorial_generate_messages_nodejs _ros_tutorial_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/xiang/test_ros/devel/share/ros_tutorial/msg/counterActionFeedback.msg" NAME_WE)
add_dependencies(ros_tutorial_generate_messages_nodejs _ros_tutorial_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/xiang/test_ros/devel/share/ros_tutorial/msg/counterFeedback.msg" NAME_WE)
add_dependencies(ros_tutorial_generate_messages_nodejs _ros_tutorial_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/xiang/test_ros/devel/share/ros_tutorial/msg/counterActionGoal.msg" NAME_WE)
add_dependencies(ros_tutorial_generate_messages_nodejs _ros_tutorial_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/xiang/test_ros/src/ros_tutorial/srv/add.srv" NAME_WE)
add_dependencies(ros_tutorial_generate_messages_nodejs _ros_tutorial_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/xiang/test_ros/devel/share/ros_tutorial/msg/counterGoal.msg" NAME_WE)
add_dependencies(ros_tutorial_generate_messages_nodejs _ros_tutorial_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/xiang/test_ros/devel/share/ros_tutorial/msg/counterActionResult.msg" NAME_WE)
add_dependencies(ros_tutorial_generate_messages_nodejs _ros_tutorial_generate_messages_check_deps_${_filename})

# target for backward compatibility
add_custom_target(ros_tutorial_gennodejs)
add_dependencies(ros_tutorial_gennodejs ros_tutorial_generate_messages_nodejs)

# register target for catkin_package(EXPORTED_TARGETS)
list(APPEND ${PROJECT_NAME}_EXPORTED_TARGETS ros_tutorial_generate_messages_nodejs)

### Section generating for lang: genpy
### Generating Messages
_generate_msg_py(ros_tutorial
  "/home/xiang/test_ros/devel/share/ros_tutorial/msg/counterAction.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/melodic/share/actionlib_msgs/cmake/../msg/GoalID.msg;/home/xiang/test_ros/devel/share/ros_tutorial/msg/counterActionGoal.msg;/opt/ros/melodic/share/actionlib_msgs/cmake/../msg/GoalStatus.msg;/home/xiang/test_ros/devel/share/ros_tutorial/msg/counterResult.msg;/home/xiang/test_ros/devel/share/ros_tutorial/msg/counterFeedback.msg;/home/xiang/test_ros/devel/share/ros_tutorial/msg/counterActionResult.msg;/opt/ros/melodic/share/std_msgs/cmake/../msg/Header.msg;/home/xiang/test_ros/devel/share/ros_tutorial/msg/counterActionFeedback.msg;/home/xiang/test_ros/devel/share/ros_tutorial/msg/counterGoal.msg"
  ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/ros_tutorial
)
_generate_msg_py(ros_tutorial
  "/home/xiang/test_ros/devel/share/ros_tutorial/msg/counterResult.msg"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/ros_tutorial
)
_generate_msg_py(ros_tutorial
  "/home/xiang/test_ros/devel/share/ros_tutorial/msg/counterActionFeedback.msg"
  "${MSG_I_FLAGS}"
  "/home/xiang/test_ros/devel/share/ros_tutorial/msg/counterFeedback.msg;/opt/ros/melodic/share/actionlib_msgs/cmake/../msg/GoalID.msg;/opt/ros/melodic/share/actionlib_msgs/cmake/../msg/GoalStatus.msg;/opt/ros/melodic/share/std_msgs/cmake/../msg/Header.msg"
  ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/ros_tutorial
)
_generate_msg_py(ros_tutorial
  "/home/xiang/test_ros/devel/share/ros_tutorial/msg/counterFeedback.msg"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/ros_tutorial
)
_generate_msg_py(ros_tutorial
  "/home/xiang/test_ros/devel/share/ros_tutorial/msg/counterActionGoal.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/melodic/share/actionlib_msgs/cmake/../msg/GoalID.msg;/home/xiang/test_ros/devel/share/ros_tutorial/msg/counterGoal.msg;/opt/ros/melodic/share/std_msgs/cmake/../msg/Header.msg"
  ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/ros_tutorial
)
_generate_msg_py(ros_tutorial
  "/home/xiang/test_ros/devel/share/ros_tutorial/msg/counterGoal.msg"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/ros_tutorial
)
_generate_msg_py(ros_tutorial
  "/home/xiang/test_ros/devel/share/ros_tutorial/msg/counterActionResult.msg"
  "${MSG_I_FLAGS}"
  "/home/xiang/test_ros/devel/share/ros_tutorial/msg/counterResult.msg;/opt/ros/melodic/share/actionlib_msgs/cmake/../msg/GoalID.msg;/opt/ros/melodic/share/actionlib_msgs/cmake/../msg/GoalStatus.msg;/opt/ros/melodic/share/std_msgs/cmake/../msg/Header.msg"
  ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/ros_tutorial
)

### Generating Services
_generate_srv_py(ros_tutorial
  "/home/xiang/test_ros/src/ros_tutorial/srv/add.srv"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/ros_tutorial
)

### Generating Module File
_generate_module_py(ros_tutorial
  ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/ros_tutorial
  "${ALL_GEN_OUTPUT_FILES_py}"
)

add_custom_target(ros_tutorial_generate_messages_py
  DEPENDS ${ALL_GEN_OUTPUT_FILES_py}
)
add_dependencies(ros_tutorial_generate_messages ros_tutorial_generate_messages_py)

# add dependencies to all check dependencies targets
get_filename_component(_filename "/home/xiang/test_ros/devel/share/ros_tutorial/msg/counterAction.msg" NAME_WE)
add_dependencies(ros_tutorial_generate_messages_py _ros_tutorial_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/xiang/test_ros/devel/share/ros_tutorial/msg/counterResult.msg" NAME_WE)
add_dependencies(ros_tutorial_generate_messages_py _ros_tutorial_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/xiang/test_ros/devel/share/ros_tutorial/msg/counterActionFeedback.msg" NAME_WE)
add_dependencies(ros_tutorial_generate_messages_py _ros_tutorial_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/xiang/test_ros/devel/share/ros_tutorial/msg/counterFeedback.msg" NAME_WE)
add_dependencies(ros_tutorial_generate_messages_py _ros_tutorial_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/xiang/test_ros/devel/share/ros_tutorial/msg/counterActionGoal.msg" NAME_WE)
add_dependencies(ros_tutorial_generate_messages_py _ros_tutorial_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/xiang/test_ros/src/ros_tutorial/srv/add.srv" NAME_WE)
add_dependencies(ros_tutorial_generate_messages_py _ros_tutorial_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/xiang/test_ros/devel/share/ros_tutorial/msg/counterGoal.msg" NAME_WE)
add_dependencies(ros_tutorial_generate_messages_py _ros_tutorial_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/xiang/test_ros/devel/share/ros_tutorial/msg/counterActionResult.msg" NAME_WE)
add_dependencies(ros_tutorial_generate_messages_py _ros_tutorial_generate_messages_check_deps_${_filename})

# target for backward compatibility
add_custom_target(ros_tutorial_genpy)
add_dependencies(ros_tutorial_genpy ros_tutorial_generate_messages_py)

# register target for catkin_package(EXPORTED_TARGETS)
list(APPEND ${PROJECT_NAME}_EXPORTED_TARGETS ros_tutorial_generate_messages_py)



if(gencpp_INSTALL_DIR AND EXISTS ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/ros_tutorial)
  # install generated code
  install(
    DIRECTORY ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/ros_tutorial
    DESTINATION ${gencpp_INSTALL_DIR}
  )
endif()
if(TARGET std_msgs_generate_messages_cpp)
  add_dependencies(ros_tutorial_generate_messages_cpp std_msgs_generate_messages_cpp)
endif()
if(TARGET actionlib_msgs_generate_messages_cpp)
  add_dependencies(ros_tutorial_generate_messages_cpp actionlib_msgs_generate_messages_cpp)
endif()

if(geneus_INSTALL_DIR AND EXISTS ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/ros_tutorial)
  # install generated code
  install(
    DIRECTORY ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/ros_tutorial
    DESTINATION ${geneus_INSTALL_DIR}
  )
endif()
if(TARGET std_msgs_generate_messages_eus)
  add_dependencies(ros_tutorial_generate_messages_eus std_msgs_generate_messages_eus)
endif()
if(TARGET actionlib_msgs_generate_messages_eus)
  add_dependencies(ros_tutorial_generate_messages_eus actionlib_msgs_generate_messages_eus)
endif()

if(genlisp_INSTALL_DIR AND EXISTS ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/ros_tutorial)
  # install generated code
  install(
    DIRECTORY ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/ros_tutorial
    DESTINATION ${genlisp_INSTALL_DIR}
  )
endif()
if(TARGET std_msgs_generate_messages_lisp)
  add_dependencies(ros_tutorial_generate_messages_lisp std_msgs_generate_messages_lisp)
endif()
if(TARGET actionlib_msgs_generate_messages_lisp)
  add_dependencies(ros_tutorial_generate_messages_lisp actionlib_msgs_generate_messages_lisp)
endif()

if(gennodejs_INSTALL_DIR AND EXISTS ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/ros_tutorial)
  # install generated code
  install(
    DIRECTORY ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/ros_tutorial
    DESTINATION ${gennodejs_INSTALL_DIR}
  )
endif()
if(TARGET std_msgs_generate_messages_nodejs)
  add_dependencies(ros_tutorial_generate_messages_nodejs std_msgs_generate_messages_nodejs)
endif()
if(TARGET actionlib_msgs_generate_messages_nodejs)
  add_dependencies(ros_tutorial_generate_messages_nodejs actionlib_msgs_generate_messages_nodejs)
endif()

if(genpy_INSTALL_DIR AND EXISTS ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/ros_tutorial)
  install(CODE "execute_process(COMMAND \"/usr/bin/python2\" -m compileall \"${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/ros_tutorial\")")
  # install generated code
  install(
    DIRECTORY ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/ros_tutorial
    DESTINATION ${genpy_INSTALL_DIR}
  )
endif()
if(TARGET std_msgs_generate_messages_py)
  add_dependencies(ros_tutorial_generate_messages_py std_msgs_generate_messages_py)
endif()
if(TARGET actionlib_msgs_generate_messages_py)
  add_dependencies(ros_tutorial_generate_messages_py actionlib_msgs_generate_messages_py)
endif()
