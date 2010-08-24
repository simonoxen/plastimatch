##-----------------------------------------------------------------------------
##  As posted on NVidia forum
##  http://forums.nvidia.com/index.php?showtopic=97795
##  Version: Oct 5, 2009
##  Downloaded: Nov 14, 2009
##  Modified by GCS
##-----------------------------------------------------------------------------

## AMD/ATI
set(ENV_ATISTREAMSDKROOT $ENV{ATISTREAMSDKROOT})
#set(ENV_NVSDKCOMPUTE_ROOT $ENV{NVSDKCOMPUTE_ROOT})

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
    PATHS $ENV{NVSDKCOMPUTE_ROOT}/OpenCL/common/inc
    NAMES CL/cl.h OpenCL/cl.h
    )

  find_library(
    OPENCL_LIBRARY
    PATHS $ENV{NVSDKCOMPUTE_ROOT}/OpenCL/common/lib/Win32
    NAMES OpenCL
    )
endif(ENV_ATISTREAMSDKROOT)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(
  OPENCL
  DEFAULT_MSG
  OPENCL_LIBRARY OPENCL_INCLUDE_DIR
  )


# JAS 08.24.2010
# Set OPENCL_FOUND for plm_config.h
if (OPENCL_INCLUDE_DIR MATCHES "NONE")
    SET (OPENCL_FOUND false CACHE BOOL "Do we have OpenCL?")
else (OPENCL_INCLUDE_DIR MATCHES "NONE")
    if (OPENCL_LIBRARY MATCHES "NONE")
        SET (OPENCL_FOUND false CACHE BOOL "Do we have OpenCL?")
    else (OPENCL_LIBRARY MATCHES "NONE")
        SET (OPENCL_FOUND true CACHE BOOL "Do we have OpenCL?")
    endif (OPENCL_LIBRARY MATCHES "NONE")
endif (OPENCL_INCLUDE_DIR MATCHES "NONE")

if(OPENCL_FOUND)
  set(OPENCL_LIBRARIES ${OPENCL_LIBRARY})
else(OPENCL_FOUND)
  set(OPENCL_LIBRARIES)
endif(OPENCL_FOUND)

mark_as_advanced(
  OPENCL_INCLUDE_DIR
  OPENCL_LIBRARY
)
