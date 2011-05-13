# - Wrapper around FindCUDA

if (MINGW)
  # Cuda doesn't work with mingw at all
  set (CUDA_FOUND FALSE)
elseif (${CMAKE_MAJOR_VERSION}.${CMAKE_MINOR_VERSION} LESS 2.8)
  # FindCuda is included with CMake 2.8
  set (CUDA_FOUND FALSE)
else ()
  # GCS 2011.03.16
  # Make nvcc less whiny
  if (CMAKE_COMPILER_IS_GNUCC)
    set (CUDA_PROPAGATE_HOST_FLAGS OFF)
    if (CMAKE_C_FLAGS)
      set (CUDA_NVCC_FLAGS --compiler-options ${CMAKE_C_FLAGS})
    endif ()
  endif ()

  find_package (CUDA QUIET)
endif ()

SET (CUDA_FOUND ${CUDA_FOUND} CACHE BOOL "Did we find cuda?")

IF (CUDA_FOUND)
  CUDA_INCLUDE_DIRECTORIES (
    ${CMAKE_CURRENT_SOURCE_DIR}
    )
ENDIF()

# JAS 08.25.2010
#   Check to make sure nvcc has gcc-4.3 for compiling.
#   This script will modify CUDA_NVCC_FLAGS if system default is not gcc-4.3
INCLUDE (nvcc-check)

# JAS 2010.12.09
#   Build code for all known compute capabilities by default.
#   When developing, it is sometimes nice to turn this off in order
#   to speed up the build processes (since you only have 1 GPU in your machine).
SET (PLM_CUDA_ALL_DEVICES ON CACHE BOOL 
  "Generate GPU code for all compute capabilities?")
IF (PLM_CUDA_ALL_DEVICES)
  MESSAGE (STATUS "CUDA Build Level: ALL Compute Capabilities")
  SET (CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}
        -gencode arch=compute_10,code=sm_10
        -gencode arch=compute_11,code=sm_11
        -gencode arch=compute_12,code=sm_12
        -gencode arch=compute_13,code=sm_13
        -gencode arch=compute_20,code=sm_20
    )
ELSE ()
  MESSAGE (STATUS "CUDA Build Level: Build system Compute Capability ONLY!")
ENDIF ()

