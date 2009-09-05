##-----------------------------------------------------------------------------
##  Look for F2C library.
##    (Only look for f2c if fortran not found)
##-----------------------------------------------------------------------------
## http://readlist.com/lists/lists.sourceforge.net/mingw-users/0/3794.html
SET (HAVE_F2C_LIBRARY FALSE)
IF (PLM_LINK_MSVCRT)
  SET (WIN32_LIBF2C "vcf2c_msvcrt")
ELSE (PLM_LINK_MSVCRT)
  SET (WIN32_LIBF2C "vcf2c_libcmt")
ENDIF (PLM_LINK_MSVCRT)

IF(NOT F2C_DIR)
  FIND_PATH(F2C_DIR f2c.h
    PATH $ENV{F2C_DIR}
    ${CMAKE_BINARY_DIR}/../libf2c
    ${CMAKE_BINARY_DIR}/../f2c
    DOC "Root directory of f2c.")
ENDIF(NOT F2C_DIR)

MESSAGE(STATUS "Looking for f2c include file")
FIND_PATH(F2C_INCLUDE_DIR f2c.h
  PATHS 
  $ENV{F2C_INC_DIR} 
  ${F2C_DIR}
  /usr/include
  /usr/local/include
  DOC "Path to f2c include files.")

MESSAGE(STATUS "Looking for f2c library")
FIND_LIBRARY(F2C_LIBRARY
  NAMES f2c ${WIN32_LIBF2C}
  PATHS
  ${CMAKE_SOURCE_DIR}
  ${CMAKE_BINARY_DIR}
  $ENV{F2C_LIB_DIR}
  ${F2C_DIR}
  nDOC "Path to f2c library.")

IF(F2C_LIBRARY)
  SET(HAVE_F2C_LIBRARY TRUE)
ENDIF(F2C_LIBRARY)
