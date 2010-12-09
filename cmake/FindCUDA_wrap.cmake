# - Wrapper around FindCUDA

IF (MINGW)
  # Cuda doesn't work with mingw at all
  SET (CUDA_FOUND FALSE)
ELSEIF (${CMAKE_MAJOR_VERSION}.${CMAKE_MINOR_VERSION} LESS 2.8)
  # FindCuda is included with CMake 2.8
  SET (CUDA_FOUND FALSE)
ELSE (MINGW)
  FIND_PACKAGE (CUDA QUIET)
ENDIF (MINGW)

SET (CUDA_FOUND ${CUDA_FOUND} CACHE BOOL "Did we find cuda?")

IF (CUDA_FOUND)
  CUDA_INCLUDE_DIRECTORIES (
    ${CMAKE_CURRENT_SOURCE_DIR}
    )
ENDIF(CUDA_FOUND)

# JAS 08.25.2010
#   Check to make sure nvcc has gcc-4.3 for compiling.
#   This script will modify CUDA_NVCC_FLAGS if system default is not gcc-4.3
INCLUDE (nvcc-check)

# JAS 2010.12.09
#   Build code for all known compute capabilities only if desired.
SET (PLM_CUDA_ALL_DEVICES OFF CACHE BOOL "Generate GPU code for all compute capabilities?")

IF (PLM_CUDA_ALL_DEVICES)
    MESSAGE (STATUS "CUDA Build Level: ALL Compute Capabilities")
    SET (CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}
        -gencode arch=compute_10,code=sm_10
        -gencode arch=compute_11,code=sm_11
        -gencode arch=compute_12,code=sm_12
        -gencode arch=compute_13,code=sm_13
        -gencode arch=compute_20,code=sm_20
    )
ELSE (PLM_CUDA_ALL_DEVICES)
    MESSAGE (STATUS "CUDA Build Level: Build system Compute Capability ONLY!")
ENDIF (PLM_CUDA_ALL_DEVICES)
