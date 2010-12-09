# - Find Slicer
# Find the Slicer includes and library
#
# The user should set Slicer_DIR to the directory containing either 
# Slicer3Config.cmake or SlicerConfig.cmake
#
#  SLICER_FOUND       - true if found

IF (SLICER_INCLUDE_DIR)
  # Already in cache, be silent
  SET (Slicer_FIND_QUIETLY TRUE)
ENDIF (SLICER_INCLUDE_DIR)

# Find SlicerConfig.cmake or Slicer3Config.cmake
UNSET (SLICER_CONFIG_FILE CACHE)
IF (Slicer_DIR OR Slicer3_DIR)
  FIND_FILE (SLICER_CONFIG_FILE SlicerConfig.cmake
    "${Slicer_DIR}" "${Slicer3_DIR}")
  IF (SLICER_CONFIG_FILE)
    SET (SLICER_IS_SLICER3 FALSE)
  ELSE (SLICER_CONFIG_FILE)
    FIND_FILE (SLICER_CONFIG_FILE Slicer3Config.cmake
      "${Slicer_DIR}" "${Slicer3_DIR}")
    IF (SLICER_CONFIG_FILE)
      SET (SLICER_IS_SLICER3 TRUE)
    ENDIF (SLICER_CONFIG_FILE)
  ENDIF (SLICER_CONFIG_FILE)
ENDIF (Slicer_DIR OR Slicer3_DIR)

# This sets Slicer_DIR in the cache.  I couldn't find any other way to do it.
IF (SLICER_CONFIG_FILE)
  GET_FILENAME_COMPONENT (SLICER_CONFIG_FILE_DIR
    "${SLICER_CONFIG_FILE}" PATH)
  SET (Slicer_DIR "${SLICER_CONFIG_FILE_DIR}"
    CACHE PATH "Directory with SlicerConfig.cmake or Slicer3Config.cmake"
    FORCE)
ELSE (SLICER_CONFIG_FILE)
  SET (Slicer_DIR "Slicer_DIR-NOTFOUND"
    CACHE PATH "Directory with SlicerConfig.cmake or Slicer3Config.cmake"
    FORCE)
ENDIF (SLICER_CONFIG_FILE)

MESSAGE (STATUS "BUILD_SHARED_LIBS: ${BUILD_SHARED_LIBS}")
IF (SLICER_CONFIG_FILE)
  INCLUDE ("${SLICER_CONFIG_FILE}")
ENDIF (SLICER_CONFIG_FILE)

# handle the QUIETLY and REQUIRED arguments and set SLICER_FOUND to TRUE if 
# all listed variables are TRUE
INCLUDE (FindPackageHandleStandardArgs)
FIND_PACKAGE_HANDLE_STANDARD_ARGS (SLICER DEFAULT_MSG Slicer_DIR)

## Convert old names to new names
IF (SLICER_FOUND AND SLICER_IS_SLICER3)
  SET (Slicer_Base_INCLUDE_DIRS "${Slicer3_Base_INCLUDE_DIRS}")
  SET (Slicer_Libs_INCLUDE_DIRS "${Slicer3_Libs_INCLUDE_DIRS}")
  SET (Slicer_Base_LIBRARIES "${Slicer3_Base_LIBRARIES}")
  SET (Slicer_Libs_LIBRARIES "${Slicer3_Libs_LIBRARIES}")
  SET (Slicer_USE_FILE "${Slicer3_USE_FILE}")
  SET (BUILD_SHARED_LIBS ON)
ENDIF (SLICER_FOUND AND SLICER_IS_SLICER3)
MESSAGE (STATUS "BUILD_SHARED_LIBS: ${BUILD_SHARED_LIBS}")

MARK_AS_ADVANCED (SLICER_CONFIG_FILE)
