# - Find Slicer
# Find the Slicer includes and library
#
# The user should set Slicer_DIR to the directory containing either 
# Slicer3Config.cmake or SlicerConfig.cmake
#
#  SLICER_FOUND       - true if found
#  SLICER_IS_SLICER3  - true if we have Slicer vesion 3.X
#  SLICER_IS_SLICER4  - true if we have Slicer vesion 4.X

if (SLICER_INCLUDE_DIR)
  # Already in cache, be silent
  set (Slicer_FIND_QUIETLY TRUE)
endif ()

# Suppress Slicer4 strangeness
set (Slicer_SKIP_PROJECT_COMMAND TRUE)
set (Slicer_SKIP_SET_CMAKE_C_CXX_FLAGS TRUE)

# Find SlicerConfig.cmake or Slicer3Config.cmake
if (NOT ${CMAKE_MAJOR_VERSION}.${CMAKE_MINOR_VERSION}.${CMAKE_PATCH_VERSION} VERSION_LESS 2.8.0)
  unset (SLICER_CONFIG_FILE CACHE)
endif ()
if (Slicer_DIR OR Slicer3_DIR)
  find_file (SLICER_CONFIG_FILE SlicerConfig.cmake
    "${Slicer_DIR}" "${Slicer3_DIR}" "${Slicer_DIR}/Slicer-build")
  if (SLICER_CONFIG_FILE)
    set (SLICER_IS_SLICER3 FALSE)
    set (SLICER_IS_SLICER4 TRUE)
  else ()
    find_file (SLICER_CONFIG_FILE Slicer3Config.cmake
      "${Slicer_DIR}" "${Slicer3_DIR}")
    if (SLICER_CONFIG_FILE)
      set (SLICER_IS_SLICER3 TRUE)
      set (SLICER_IS_SLICER4 FALSE)
    endif ()
  endif ()
endif ()

# This sets Slicer_DIR in the cache.  I couldn't find any other way to do it.
if (SLICER_CONFIG_FILE)
  get_filename_component (SLICER_CONFIG_FILE_DIR
    "${SLICER_CONFIG_FILE}" PATH)
  set (Slicer_DIR "${SLICER_CONFIG_FILE_DIR}"
    CACHE PATH "Directory with SlicerConfig.cmake or Slicer3Config.cmake"
    FORCE)
else ()
  set (Slicer_DIR "Slicer_DIR-NOTFOUND"
    CACHE PATH "Directory with SlicerConfig.cmake or Slicer3Config.cmake"
    FORCE)
endif ()

message (STATUS "BUILD_SHARED_LIBS: ${BUILD_SHARED_LIBS}")
message (STATUS "BUILD_AGAINST_SLICER3: ${BUILD_AGAINST_SLICER3}")
if (SLICER_CONFIG_FILE)
  # The Slicer4 config file complains if these are set.
  unset (ITK_DIR CACHE)
  unset (QT_QMAKE_EXECUTABLE CACHE)
  unset (VTK_DIR CACHE)
  include ("${SLICER_CONFIG_FILE}")
endif ()

# handle the QUIETLY and REQUIRED arguments and set SLICER_FOUND to TRUE if 
# all listed variables are TRUE
include (FindPackageHandleStandardArgs)
find_package_handle_standard_args (SLICER DEFAULT_MSG Slicer_DIR)

if (SLICER_FOUND)

  # For Slicer 3: convert old names to new names
  if (SLICER_IS_SLICER3)
    set (Slicer_Base_INCLUDE_DIRS "${Slicer3_Base_INCLUDE_DIRS}")
    set (Slicer_Libs_INCLUDE_DIRS "${Slicer3_Libs_INCLUDE_DIRS}")
    set (Slicer_Base_LIBRARIES "${Slicer3_Base_LIBRARIES}")
    set (Slicer_Libs_LIBRARIES "${Slicer3_Libs_LIBRARIES}")
    set (Slicer_USE_FILE "${Slicer3_USE_FILE}")
  endif ()

  # Slicer 4 requires GIT_EXECUTABLE to be defined.
  if (SLICER_IS_SLICER4)
    if (NOT DEFINED GIT_EXECUTABLE)
      find_package (Git)
    endif ()
    if (NOT GIT_FOUND)
      find_program(GIT_EXECUTABLE
	NAMES git
	PATHS c:/cygwin/bin
	DOC "git command line client"
	)
    endif ()
  endif ()

  ## Always set shared libs on
  set (BUILD_SHARED_LIBS ON)
endif ()
message (STATUS "BUILD_SHARED_LIBS: ${BUILD_SHARED_LIBS}")

mark_as_advanced (SLICER_CONFIG_FILE)
