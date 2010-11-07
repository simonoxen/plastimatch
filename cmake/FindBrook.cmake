######################################################
##  RULES FOR FINDING BROOK
##  These variables are set:
##    BROOK_FOUND
##    BROOK_INCLUDE_DIR
##    BROOK_LIBRARIES
##    BRCC_EXECUTABLE
######################################################
SET(BROOK_FOUND FOOBAR)

IF(NOT BROOKDIR)
  FIND_PATH(BROOKDIR common.mk
    PATH $ENV{BROOKDIR}
    DOC "Root directory of brook.")
ENDIF(NOT BROOKDIR)

MESSAGE(STATUS "Looking for brcc")
IF(NOT BRCC_EXECUTABLE)
  FIND_PROGRAM(BRCC_EXECUTABLE
               NAMES brcc
               PATHS ${BROOKDIR}/bin $ENV{BROOKDIR}/bin
	       DOC "Path to brcc executable.")
ENDIF(NOT BRCC_EXECUTABLE)

MESSAGE(STATUS "Looking for brook includes")
FIND_PATH(BROOK_INCLUDE_DIR brook/brook.hpp
          PATHS ${BROOKDIR}/include $ENV{BROOKDIR}/include
          DOC "Path to brook include directory")

MESSAGE(STATUS "Looking for brook libraries")
FIND_LIBRARY(BROOK_LIBRARIES
  NAMES brook
  PATHS ${BROOKDIR}/bin $ENV{BROOKDIR}/bin
  DOC "Path to brook library.")

IF(BRCC_EXECUTABLE AND BROOK_INCLUDE_DIR AND BROOK_LIBRARIES)
  SET(BROOK_FOUND TRUE)
ELSE(BRCC_EXECUTABLE AND BROOK_INCLUDE_DIR AND BROOK_LIBRARIES)
  SET(BROOK_FOUND FALSE)
ENDIF(BRCC_EXECUTABLE AND BROOK_INCLUDE_DIR AND BROOK_LIBRARIES)

MARK_AS_ADVANCED(
  BRCC_EXECUTABLE
  BROOK_INCLUDE_DIR
  BROOK_LIBRARIES)

IF(BROOK_FOUND)
  MACRO(BRCC_FILE FILENAME)
    GET_FILENAME_COMPONENT(PATH "${FILENAME}" PATH)
    GET_FILENAME_COMPONENT(HEAD "${FILENAME}" NAME_WE)
    IF(NOT EXISTS "${CMAKE_CURRENT_BINARY_DIR}/${PATH}")
      FILE(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/${PATH}")
    ENDIF(NOT EXISTS "${CMAKE_CURRENT_BINARY_DIR}/${PATH}")
    SET(OUTFILE_BASE "${CMAKE_CURRENT_BINARY_DIR}/${PATH}/${HEAD}")
    SET(OUTFILE "${OUTFILE_BASE}.cpp")

#    ADD_CUSTOM_COMMAND(
#        OUTPUT "${OUTFILE}"
#        COMMAND "${BRCC_EXECUTABLE}"
##        ARGS "-o" "${OUTFILE_BASE}" "-k" "-p" "fp40" "-p" "ps30" 
#        ARGS "-o" "${OUTFILE_BASE}" "-k" "-p" "ps30"
#        "${CMAKE_CURRENT_SOURCE_DIR}/${FILENAME}"
#        DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/${FILENAME}")

    ADD_CUSTOM_COMMAND(
        OUTPUT "${OUTFILE}"
        COMMAND ${CMAKE_COMMAND} 
	-DBRCC_EXECUTABLE="${BRCC_EXECUTABLE}"
	-DOUTFILE_BASE="${OUTFILE_BASE}"
	-DINFILE="${CMAKE_CURRENT_SOURCE_DIR}/${FILENAME}"
	-P ${CMAKE_CURRENT_SOURCE_DIR}/RUN_BRCC.cmake
        DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/${FILENAME}")
    SET_SOURCE_FILES_PROPERTIES("${OUTFILE}" PROPERTIES GENERATED TRUE)
  ENDMACRO(BRCC_FILE)
ENDIF(BROOK_FOUND)

######################################################
##  DIRECTX
##  These variables are set:
##    DIRECTX_FOUND
##    DIRECTX_LIBRARIES
##	    "C:/Program Files/Microsoft DirectX SDK (February 2007)"
######################################################
SET(DIRECTX_FOUND FALSE)
IF(WIN32)
  FIND_PATH(DIRECTX_ROOT_DIR "Include/dinput.h"
            $ENV{DXSDK_DIR}
            "C:/Program Files/Microsoft DirectX SDK (June 2007)"
            "C:/Program Files/Microsoft DirectX SDK (August 2006)"
            "C:/Program Files/Microsoft DirectX 9.0 SDK (June 2005)"
            "C:/Program Files/Microsoft DirectX 9.0 SDK (Summer 2004)"
	    )
  FIND_LIBRARY(D3DX9_LIBRARY
    NAMES d3dx9
    PATHS
      ${DIRECTX_ROOT_DIR}/Lib
      ${DIRECTX_ROOT_DIR}/Lib/x86
    DOC "Path to DirectX library.")
  FIND_LIBRARY(D3D9_LIBRARY
    NAMES d3d9
    PATHS
      ${DIRECTX_ROOT_DIR}/Lib
      ${DIRECTX_ROOT_DIR}/Lib/x86
    DOC "Path to DirectX library.")
  SET(DIRECTX_LIBRARIES ${D3DX9_LIBRARY} ${D3D9_LIBRARY})
  IF(DIRECTX_ROOT_DIR)
    SET(DIRECTX_FOUND TRUE)
  ENDIF(DIRECTX_ROOT_DIR)
ENDIF(WIN32)

MARK_AS_ADVANCED(
  D3DX9_LIBRARY
  D3D9_LIBRARY)

IF (BROOK_FOUND)
  MESSAGE (STATUS "Looking for brook - found.")
ELSE (BROOK_FOUND)
  MESSAGE (STATUS "Looking for brook - not found.")
ENDIF (BROOK_FOUND)
