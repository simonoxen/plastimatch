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

# JAS 2010.11.23
# Make life easier for people who only
# want to build plastimatch components
# that don't depend on ITK.
FIND_PACKAGE (ITK)

IF (NOT ITK_FOUND)
    MESSAGE (STATUS "ITK not found, omitting RTK")
ELSE(NOT ITK_FOUND)
    MESSAGE (STATUS "ITK found, including RTK")
    ADD_SUBDIRECTORY (${RTK_SOURCE_DIR})
ENDIF (NOT ITK_FOUND)

IF (NOT RTK_DIR)
  FIND_PATH (RTK_DIR RTKconfig.cmake
    $ENV{RTK_DIR}
    DOC "directory containing RTK build files")
ENDIF (NOT RTK_DIR)

IF (RTK_DIR AND ITK_FOUND)
  SET (RTK_FOUND 1)
  INCLUDE (${RTK_DIR}/RTKconfig.cmake)
ENDIF (RTK_DIR AND ITK_FOUND)
