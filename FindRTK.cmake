# - Find RTK
# Find the RTK includes and library
#
#  RTK_INCLUDE_DIR - where to find RTK header files
#  RTK_LIBRARIES   - List of libraries when using RTK
#  RTK_FOUND       - True if RTK found

IF (NOT RTK_DIR)
  FIND_PATH (RTK_DIR RTKconfig.cmake
    $ENV{RTK_DIR}
    DOC "directory containing RTK build files")
ENDIF (NOT RTK_DIR)

IF (RTK_DIR)
  SET (RTK_FOUND 1)
  INCLUDE (${RTK_DIR}/RTKconfig.cmake)
ELSE (RTK_DIR)
  ADD_SUBDIRECTORY (libs/RTK)
ENDIF (RTK_DIR)
