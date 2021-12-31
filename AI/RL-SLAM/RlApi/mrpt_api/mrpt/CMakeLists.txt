PROJECT(mrpt)

# -----------------------------------------------
# 1) Enumerate al subdirectories under "otherlibs/mrpt"
# -----------------------------------------------
list_subdirectories(LIST_MRPT_APPS_DIRS "${TOP_LEVEL_SRC_DIR}/otherlibs/mrpt")

# -----------------------------------------------
# 2) Add all subdirectories in makefile
# -----------------------------------------------
FOREACH(dir ${LIST_MRPT_APPS_DIRS})
	ADD_SUBDIRECTORY(${dir})
ENDFOREACH(dir)

