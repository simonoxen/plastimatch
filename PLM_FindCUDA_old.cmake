######################################################
##  RULES FOR FINDING CUDA
##  These variables are set:
##    CUDA_FOUND
##    CUDA_INCLUDE_DIR
##    CUDA_SDK_INCLUDE_DIR
##    CUDA_LIBRARIES
##    NVCC_EXECUTABLE
##  This is only working for CUDA 2.0...
######################################################
SET(CUDA_FOUND FOOBAR)

MESSAGE(STATUS "Looking for CUDA include directory.")
IF(NOT CUDA_INCLUDE_DIR)
  FIND_PATH(CUDA_INCLUDE_DIR cuda.h
    PATHS $ENV{CUDA_INC_DIR} C:/CUDA/include
    DOC "Path to cuda include files.")
ENDIF(NOT CUDA_INCLUDE_DIR)

IF(NOT CUDA_SDK_INCLUDE_DIR)
  FIND_PATH(CUDA_SDK_INCLUDE_DIR cutil.h
    PATHS $ENV{CUDA_SDK_INC_DIR} 
    "C:/Program Files/NVIDIA Corporation/NVIDIA CUDA SDK/common/inc"
    "C:/ProgramData/NVIDIA Corporation/NVIDIA CUDA SDK/common/inc"
    DOC "Path to cuda sdk include files.")
ENDIF(NOT CUDA_SDK_INCLUDE_DIR)

IF(NOT CUDA_SDK_LIB_DIR)
  FIND_PATH(CUDA_SDK_LIB_DIR cutil32.lib
    PATHS $ENV{CUDA_SDK_LIB_DIR}
    "C:/Program Files/NVIDIA Corporation/NVIDIA CUDA SDK/common/lib"
    "C:/ProgramData/NVIDIA Corporation/NVIDIA CUDA SDK/common/lib"
    DOC "Path to cuda sdk library files.")
ENDIF(NOT CUDA_SDK_LIB_DIR)

MESSAGE(STATUS "Looking for CUDA libraries.")
FIND_LIBRARY(CUDA_LIBRARY
  NAMES cuda
  PATHS $ENV{CUDA_LIB_PATH} C:/CUDA/lib
  DOC "Path to cuda library.")
FIND_LIBRARY(CUDART_LIBRARY
  NAMES cudart
  PATHS $ENV{CUDA_LIB_PATH} C:/CUDA/lib
  DOC "Path to cudart library.")
FIND_LIBRARY(CUTIL32_LIBRARY
  NAMES cutil32
  PATHS $ENV{CUDA_SDK_LIB_DIR} 
  "C:/Program Files/NVIDIA Corporation/NVIDIA CUDA SDK/common/lib"
    "C:/ProgramData/NVIDIA Corporation/NVIDIA CUDA SDK/common/lib"
  DOC "Path to cutil32 library.")
SET(CUDA_LIBRARIES ${CUDA_LIBRARY} ${CUDART_LIBRARY} ${CUTIL32_LIBRARY})

MESSAGE(STATUS "Looking for nvcc.")
IF(NOT NVCC_EXECUTABLE)
  FIND_PROGRAM(NVCC_EXECUTABLE
               NAMES nvcc
               PATHS ${CUDA_BIN_PATH} C:/CUDA/bin
	       DOC "Path to nvcc executable.")
ENDIF(NOT NVCC_EXECUTABLE)

IF(NVCC_EXECUTABLE AND CUDA_INCLUDE_DIR AND CUDA_SDK_INCLUDE_DIR AND CUDA_LIBRARIES)
  SET(CUDA_FOUND TRUE)
ELSE(NVCC_EXECUTABLE AND CUDA_INCLUDE_DIR AND CUDA_SDK_INCLUDE_DIR AND CUDA_LIBRARIES)
  SET(CUDA_FOUND FALSE)
ENDIF(NVCC_EXECUTABLE AND CUDA_INCLUDE_DIR AND CUDA_SDK_INCLUDE_DIR AND CUDA_LIBRARIES)

## Mingw isn't supported by mingw/gcc.  However, maybe this works:
## http://forums.nvidia.com/index.php?showtopic=99096
## or possibly this:
## http://forums.nvidia.com/lofiversion/index.php?t61416.html
IF(MINGW)
  SET(CUDA_FOUND FALSE)
ENDIF(MINGW)

IF(CUDA_FOUND)
  MACRO(NVCC_FILE FILENAME)
    GET_FILENAME_COMPONENT(PATH "${FILENAME}" PATH)
    GET_FILENAME_COMPONENT(HEAD "${FILENAME}" NAME_WE)
    IF(NOT EXISTS "${CMAKE_CURRENT_BINARY_DIR}/${PATH}")
      FILE(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/${PATH}")
    ENDIF(NOT EXISTS "${CMAKE_CURRENT_BINARY_DIR}/${PATH}")
    SET(OUTFILE_BASE "${CMAKE_CURRENT_BINARY_DIR}/${PATH}/${HEAD}")
##    SET(OUTFILE "${OUTFILE_BASE}.obj")
    SET(OUTFILE "${OUTFILE_BASE}.cu.cpp")
    IF(GPUIT_LINK_MSVCRT)
      SET(XCOMPILER_OPTIONS "/EHsc,/W3,/nologo,/Wp64,/O2,/Zi,/MD")
    ELSE(GPUIT_LINK_MSVCRT)
      SET(XCOMPILER_OPTIONS "/EHsc,/W3,/nologo,/Wp64,/O2,/Zi,/MT")
    ENDIF(GPUIT_LINK_MSVCRT)
    ## This only makes a release target
    ADD_CUSTOM_COMMAND(
        OUTPUT "${OUTFILE}"
        COMMAND "${NVCC_EXECUTABLE}"
        ARGS
	"-cuda"
##	"-ccbin" clbin xxxxxx
##	"-c"
##	"-DWIN32" "-D_CONSOLE" "-D_MBCS"
	## Commenting out the following line slows FDK_CUDA by an order of magnitude!!
	"-Xcompiler" "/EHsc,/W3,/nologo,/Wp64,/O2,/Zi,/MT"
##	"-Xcompiler" "/EHsc,/W3,/nologo,/Wp64,/O2,/Zi,/MD"
##	"-Xcompiler" ${XCOMPILER_OPTIONS}
	"-I" ${CMAKE_CURRENT_SOURCE_DIR}
	"-I" ${CUDA_SDK_INCLUDE_DIR}
	## PLM_CUDA_COMPILE tells the plastimatch config file not to 
	## include the itk header file
	"-DPLM_CUDA_COMPILE"
	"-o" "${OUTFILE}"
        "${CMAKE_CURRENT_SOURCE_DIR}/${FILENAME}"
        DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/${FILENAME}")
    SET_SOURCE_FILES_PROPERTIES("${OUTFILE}" PROPERTIES GENERATED TRUE)
  ENDMACRO(NVCC_FILE)
ENDIF(CUDA_FOUND)

