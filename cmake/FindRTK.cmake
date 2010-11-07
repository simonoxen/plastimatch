# - Find RTK
# Find the RTK includes and library
#
#  RTK_SOURCE_DIR  - where to find RTK source directory

IF (NOT RTK_SOURCE_DIR)
  FIND_PATH (RTK_SOURCE_DIR RTKconfig.cmake.in
    "${CMAKE_SOURCE_DIR}/libs/RTK.git"
    "${CMAKE_SOURCE_DIR}/libs/RTK"
    "${CMAKE_SOURCE_DIR}/libs/RTK.svn"
    $ENV{RTK_SOURCE_DIR}
    DOC "directory containing RTK source files")
ENDIF (NOT RTK_SOURCE_DIR)

IF (NOT RTK_SOURCE_DIR)
  MESSAGE (ERROR "Sorry, I couldn't find the RTK directory")
ENDIF (NOT RTK_SOURCE_DIR)

ADD_SUBDIRECTORY (${RTK_SOURCE_DIR})

IF (NOT RTK_DIR)
  FIND_PATH (RTK_DIR RTKconfig.cmake
    $ENV{RTK_DIR}
    DOC "directory containing RTK build files")
ENDIF (NOT RTK_DIR)

IF (RTK_DIR)
  SET (RTK_FOUND 1)
  INCLUDE (${RTK_DIR}/RTKconfig.cmake)
ENDIF (RTK_DIR)
