# - find DCMTK libraries
#

#  DCMTK_INCLUDE_DIR   - Directories to include to use DCMTK
#  DCMTK_LIBRARIES     - Files to link against to use DCMTK
#  DCMTK_FOUND         - If false, don't try to use DCMTK
#  DCMTK_DIR           - (optional) Source directory for DCMTK
#
# DCMTK_DIR can be used to make it simpler to find the various include
# directories and compiled libraries if you've just compiled it in the
# source tree. Just set it to the root of the tree where you extracted
# the source.
#
# Written for VXL by Amitha Perera.
# 
# On debian, require the following packages:
#   libdcmtk1-dev
#   libpng12-dev
#   libtiff4-dev
#   libwrap0-dev

INCLUDE(CheckLibraryExists)
INCLUDE(FindThreads)

FIND_PACKAGE (ZLIB)
IF (ZLIB_FOUND)
  MESSAGE (STATUS "Looking for ZLIB - found")
ELSE (ZLIB_FOUND)
  MESSAGE (STATUS "Looking for ZLIB - not found")
ENDIF (ZLIB_FOUND)

FIND_PACKAGE (PNG)
IF (PNG_FOUND)
  MESSAGE (STATUS "Looking for PNG - found")
ELSE (PNG_FOUND)
  MESSAGE (STATUS "Looking for PNG - not found")
ENDIF (PNG_FOUND)

FIND_PACKAGE (TIFF)
IF (TIFF_FOUND)
  MESSAGE (STATUS "Looking for TIFF - found")
ELSE (TIFF_FOUND)
  MESSAGE (STATUS "Looking for TIFF - not found")
ENDIF (TIFF_FOUND)

FIND_LIBRARY (SSL_LIBRARY ssl)
IF (SSL_LIBRARY)
  MESSAGE (STATUS "Looking for SSL - found: ${SSL_LIBRARY}")
ELSE (SSL_LIBRARY)
  MESSAGE (STATUS "Looking for SSL - not found")
ENDIF (SSL_LIBRARY)

FIND_PATH (
  DCMTK_INCLUDE_DIR dcmtk/config/osconfig.h
  ${DCMTK_DIR}/include
  )

#FIND_PATH( DCMTK_config_INCLUDE_DIR osconfig.h
#  ${DCMTK_DIR}/config/include
#  ${DCMTK_DIR}/include
#  ${DCMTK_DIR}/include/dcmtk/config
#)

#FIND_PATH( DCMTK_ofstd_INCLUDE_DIR ofstdinc.h
#  ${DCMTK_DIR}/ofstd/include
#  ${DCMTK_DIR}/include/ofstd
#  ${DCMTK_DIR}/include/dcmtk/ofstd
#)

FIND_LIBRARY(DCMTK_ofstd_LIBRARY ofstd
  ${DCMTK_DIR}/ofstd/libsrc
  ${DCMTK_DIR}/ofstd/libsrc/Release
  ${DCMTK_DIR}/ofstd/libsrc/Debug
  ${DCMTK_DIR}/ofstd/Release
  ${DCMTK_DIR}/ofstd/Debug
  ${DCMTK_DIR}/lib
)

#FIND_PATH( DCMTK_dcmdata_INCLUDE_DIR dctypes.h
#  ${DCMTK_DIR}/dcmdata/include
#  ${DCMTK_DIR}/include/dcmdata
#  ${DCMTK_DIR}/include/dcmtk/dcmdata
#)

FIND_LIBRARY( DCMTK_dcmdata_LIBRARY dcmdata
  ${DCMTK_DIR}/dcmdata/libsrc
  ${DCMTK_DIR}/dcmdata/libsrc/Release
  ${DCMTK_DIR}/dcmdata/libsrc/Debug
  ${DCMTK_DIR}/dcmdata/Release
  ${DCMTK_DIR}/dcmdata/Debug
  ${DCMTK_DIR}/lib
)


#FIND_PATH( DCMTK_dcmimgle_INCLUDE_DIR dcmimage.h
#  ${DCMTK_DIR}/dcmimgle/include
#  ${DCMTK_DIR}/include/dcmimgle
#  ${DCMTK_DIR}/include/dcmtk/dcmimgle
#)

FIND_LIBRARY( DCMTK_dcmimgle_LIBRARY dcmimgle
  ${DCMTK_DIR}/dcmimgle/libsrc
  ${DCMTK_DIR}/dcmimgle/libsrc/Release
  ${DCMTK_DIR}/dcmimgle/libsrc/Debug
  ${DCMTK_DIR}/dcmimgle/Release
  ${DCMTK_DIR}/dcmimgle/Debug
  ${DCMTK_DIR}/lib
)

FIND_LIBRARY(DCMTK_imagedb_LIBRARY 
  NAMES imagedb dcmimage
  PATHS
  ${DCMTK_DIR}/imagectn/libsrc/Release
  ${DCMTK_DIR}/imagectn/libsrc/
  ${DCMTK_DIR}/imagectn/libsrc/Debug
  ${DCMTK_DIR}/lib/
)

FIND_LIBRARY(DCMTK_dcmnet_LIBRARY dcmnet 
  ${DCMTK_DIR}/dcmnet/libsrc/Release
  ${DCMTK_DIR}/dcmnet/libsrc/Debug
  ${DCMTK_DIR}/dcmnet/libsrc/
  ${DCMTK_DIR}/lib/
)

FIND_LIBRARY(DCMTK_dcmtls_LIBRARY dcmtls 
  ${DCMTK_DIR}/dcmnet/libsrc/Release
  ${DCMTK_DIR}/dcmnet/libsrc/Debug
  ${DCMTK_DIR}/dcmnet/libsrc/
  ${DCMTK_DIR}/lib/
)


#IF( DCMTK_config_INCLUDE_DIR 
#    AND DCMTK_ofstd_INCLUDE_DIR 
#    AND DCMTK_ofstd_LIBRARY
#    AND DCMTK_dcmdata_INCLUDE_DIR
#    AND DCMTK_dcmdata_LIBRARY
#    AND DCMTK_dcmimgle_INCLUDE_DIR
#    AND DCMTK_dcmimgle_LIBRARY )

IF(DCMTK_INCLUDE_DIR 
    AND DCMTK_ofstd_LIBRARY
    AND DCMTK_dcmdata_LIBRARY
    AND DCMTK_dcmimgle_LIBRARY)

  SET( DCMTK_FOUND "YES" )

#  SET( DCMTK_INCLUDE_DIR
#    ${DCMTK_config_INCLUDE_DIR}
#    ${DCMTK_ofstd_INCLUDE_DIR}
#    ${DCMTK_dcmdata_INCLUDE_DIR}
#    ${DCMTK_dcmimgle_INCLUDE_DIR}
#  )

  SET(DCMTK_LIBRARIES
    ${DCMTK_ofstd_LIBRARY}
    ${DCMTK_dcmimgle_LIBRARY}
    ${DCMTK_dcmdata_LIBRARY}
    ${DCMTK_config_LIBRARY}
  )

  IF(DCMTK_imagedb_LIBRARY)
   SET(DCMTK_LIBRARIES
     ${DCMTK_LIBRARIES}
     ${DCMTK_imagedb_LIBRARY}
     )
  ENDIF(DCMTK_imagedb_LIBRARY)

  IF(DCMTK_dcmtls_LIBRARY)
   SET(DCMTK_LIBRARIES
     ${DCMTK_LIBRARIES}
     ${DCMTK_dcmtls_LIBRARY}
     )
   IF (SSL_LIBRARY)
   SET(DCMTK_LIBRARIES
     ${DCMTK_LIBRARIES}
     ${SSL_LIBRARY}
     )
   ENDIF (SSL_LIBRARY)
  ENDIF(DCMTK_dcmtls_LIBRARY)

  IF(DCMTK_dcmnet_LIBRARY)
   SET(DCMTK_LIBRARIES
     ${DCMTK_LIBRARIES}
     ${DCMTK_dcmnet_LIBRARY})
  ENDIF(DCMTK_dcmnet_LIBRARY)

  IF(PNG_FOUND)
    SET(DCMTK_LIBRARIES
      ${DCMTK_LIBRARIES}
      ${PNG_LIBRARIES})
  ENDIF(PNG_FOUND)

  IF(TIFF_FOUND)
    SET(DCMTK_LIBRARIES
      ${DCMTK_LIBRARIES}
      ${TIFF_LIBRARIES})
  ENDIF(TIFF_FOUND)

  IF(ZLIB_FOUND)
    SET(DCMTK_LIBRARIES
      ${DCMTK_LIBRARIES}
      ${ZLIB_LIBRARIES})
  ENDIF(ZLIB_FOUND)

  IF(CMAKE_THREAD_LIBS_INIT)
    SET(DCMTK_LIBRARIES
      ${DCMTK_LIBRARIES}
      ${CMAKE_THREAD_LIBS_INIT})
  ENDIF(CMAKE_THREAD_LIBS_INIT)

  FIND_LIBRARY(LIBWRAP_LIBRARY NAMES wrap libwrap PATHS /lib)
  IF(LIBWRAP_LIBRARY)
    SET(DCMTK_LIBRARIES
      ${DCMTK_LIBRARIES}
      ${LIBWRAP_LIBRARY})
  ENDIF(LIBWRAP_LIBRARY)

  IF(WIN32)
    SET(DCMTK_LIBRARIES ${DCMTK_LIBRARIES} netapi32 ws2_32)
  ENDIF(WIN32)

#ENDIF( DCMTK_config_INCLUDE_DIR 
#    AND DCMTK_ofstd_INCLUDE_DIR 
#    AND DCMTK_ofstd_LIBRARY
#    AND DCMTK_dcmdata_INCLUDE_DIR
#    AND DCMTK_dcmdata_LIBRARY
#    AND DCMTK_dcmimgle_INCLUDE_DIR
#    AND DCMTK_dcmimgle_LIBRARY )

ENDIF(DCMTK_INCLUDE_DIR 
    AND DCMTK_ofstd_LIBRARY
    AND DCMTK_dcmdata_LIBRARY
    AND DCMTK_dcmimgle_LIBRARY)

IF( NOT DCMTK_FOUND )
  SET( DCMTK_DIR "" CACHE PATH "Root of DCMTK source tree (optional)." )
#  MARK_AS_ADVANCED( DCMTK_DIR )
ENDIF( NOT DCMTK_FOUND )

IF (DCMTK_FOUND)
  MESSAGE (STATUS "Looking for dcmtk - found.")
ELSE (DCMTK_FOUND)
  MESSAGE (STATUS "Looking for dcmtk - not found.")
ENDIF (DCMTK_FOUND)
