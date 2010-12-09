##-----------------------------------------------------------------------------
##  As posted on NVidia forum
##  http://forums.nvidia.com/index.php?showtopic=97795
##  Version: Oct 5, 2009
##  Downloaded: Nov 14, 2009
##  Modified by GCS
##-----------------------------------------------------------------------------

# JAS 2010.12.09
# Option to not link against OpenCL libraries (for Linux)
# This prevents ld from attempting to load libOpenCL.so upon execution,
# which allows us do delay loading instead.  Only needed for Linux
# The behavior of this is very similar to PLM_USE_CUDA_PLUGIN
SET (PLM_OPENCL_DELAY OFF CACHE BOOL "Enable delay loading of OpenCL")

## AMD/ATI
set(ENV_ATISTREAMSDKROOT $ENV{ATISTREAMSDKROOT})

if(ENV_ATISTREAMSDKROOT)
  find_path(
    OPENCL_INCLUDE_DIR
    NAMES CL/cl.h OpenCL/cl.h
    PATHS $ENV{ATISTREAMSDKROOT}/include
    NO_DEFAULT_PATH
    )

  ## Both windows and linux follow this directory structure.  Not sure 
  ## about darwin.
  if(CMAKE_SIZEOF_VOID_P EQUAL 4)
    set(
      OPENCL_LIB_SEARCH_PATH
      ${OPENCL_LIB_SEARCH_PATH}
      $ENV{ATISTREAMSDKROOT}/lib/x86
      )
  else(CMAKE_SIZEOF_VOID_P EQUAL 4)
    set(
      OPENCL_LIB_SEARCH_PATH
      ${OPENCL_LIB_SEARCH_PATH}
      $ENV{ATISTREAMSDKROOT}/lib/x86_64
      )
  endif(CMAKE_SIZEOF_VOID_P EQUAL 4)

  find_library(
    OPENCL_LIBRARY
    NAMES OpenCL
    PATHS ${OPENCL_LIB_SEARCH_PATH}
    NO_DEFAULT_PATH
    )

## NVIDIA
else(ENV_ATISTREAMSDKROOT)
  find_path(
    OPENCL_INCLUDE_DIR
    PATHS $ENV{CUDA_INC_PATH}
    NAMES CL/cl.h OpenCL/cl.h
    )

  find_library(
    OPENCL_LIBRARY
    PATHS $ENV{CUDA_LIB_PATH}
    NAMES OpenCL
    )
endif(ENV_ATISTREAMSDKROOT)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(
  OPENCL
  DEFAULT_MSG
  OPENCL_LIBRARY OPENCL_INCLUDE_DIR
  )

# JAS 2010.12.09
# Edit to allow OpenCL to be delay loaded
IF (OPENCL_FOUND)
    IF (PLM_OPENCL_DELAY)
        IF (WIN32)
            SET (OPENCL_LIBRARIES ${OPENCL_LIBRARY})
        ELSE (WIN32)
            SET (OPENCL_LIBRARIES)
        ENDIF (WIN32)
    ELSE (PLM_OPENCL_DELAY)
        SET (OPENCL_LIBRARIES ${OPENCL_LIBRARY})
    ENDIF (PLM_OPENCL_DELAY)
ELSE (OPENCL_FOUND)
    SET (OPENCL_LIBRARIES)
ENDIF (OPENCL_FOUND)

if(MINGW)
  set(OPENCL_FOUND FALSE)
endif(MINGW)

mark_as_advanced(
  OPENCL_INCLUDE_DIR
  OPENCL_LIBRARY
)
