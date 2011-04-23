# MIL = Matrox Imaging Library
# Sets the following variables
#
#   HAVE_MIL
#   MIL_FOUND
#   MIL_INCLUDE_DIR
#   MIL_LIBRARIES
#
# Only works on Windows

SET(MIL_FOUND FALSE)
IF (WIN32)
  FIND_PATH(MIL_ROOT_DIR "include/mil.h"
    "C:/Program Files/Matrox Imaging/mil"
    )
  SET(MIL_INCLUDE_DIR $(MIL_ROOT_DIR)/include
    DOC "Path to MIL include directory")
  FIND_PATH(MIL_LIBRARY_PATH "mil.lib"
    $(MIL_ROOT_DIR)/library/winnt/msc/dll
    DOC "Path to MIL libraries.")
  FIND_LIBRARY(MIL_MIL_LIB mil ${MIL_LIBRARY_PATH})
  FIND_LIBRARY(MIL_MILVGA_LIB milvga ${MIL_LIBRARY_PATH})
  FIND_LIBRARY(MIL_MILMET2D_LIB milmet2d ${MIL_LIBRARY_PATH})
  IF(MIL_ROOT_DIR)
    SET(MIL_FOUND TRUE)
    SET(MIL_LIBRARIES ${MIL_MIL_LIB} ${MIL_MILVGA_LIB} ${MIL_MILMET2D_LIB})
  ELSE(MIL_ROOT_DIR)
    SET(MIL_LIBRARIES "")
  ENDIF(MIL_ROOT_DIR)
ENDIF(WIN32)
SET(HAVE_MIL ${MIL_FOUND})
