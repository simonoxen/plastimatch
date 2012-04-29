##-----------------------------------------------------------------------------
##  See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
##-----------------------------------------------------------------------------
project (src_plastimatch_test)

option (PLM_BUILD_CUDA_TEST "Build CUDA test program" OFF)
option (PLM_BUILD_DCMTK_TEST "Build DCMTK test program" OFF)
option (PLM_BUILD_DLIB_TEST "Build dlib test program" OFF)
option (PLM_BUILD_FANN_TEST "Build fann test program" OFF)
option (PLM_BUILD_GDCM1_TEST "Build GDCM test program" OFF)
option (PLM_BUILD_MEX_TEST "Build mex test program" OFF)
option (PLM_BUILD_NLOPT_TEST "Build NLopt test program" OFF)
option (PLM_BUILD_OPENCL_TEST "Build OpenCL test program" OFF)
option (PLM_BUILD_OPENMP_TEST "Build OpenMP test program" OFF)
option (PLM_BUILD_QT_TEST "Build Qt test program" OFF)

include_directories (BEFORE ${CMAKE_CURRENT_SOURCE_DIR})

set (CUDA_TEST_SRC
  cuda_test.cu
  )
set (DCMTK_TEST_SRC
  dcmtk_test.cxx
  )
set (TEST_EPS_SRC
  test_eps.cxx
  )
set (TEST_LIMITS_SRC
  test_limits.cxx
  )
set (TEST_OVERFLOW_SRC
  test_overflow.cxx
  )


# Test executable -- cuda
if (CUDA_FOUND AND PLM_BUILD_CUDA_TEST)
  cuda_compile (CUDA_TEST_WRAPPERS ${CUDA_TEST_SRC})
  plm_add_executable (cuda_test "${CUDA_TEST_WRAPPERS}" 
    "${CUDA_LIBRARIES}" "" 
    ${BUILD_ALWAYS} ${INSTALL_NEVER})
  set_target_properties (cuda_test PROPERTIES LINKER_LANGUAGE CXX)
endif ()

# Test executable -- dcmtk
if (DCMTK_FOUND AND DCMTK_VERSION_36 AND PLM_BUILD_DCMTK_TEST
    AND FULL_PLASTIMATCH_BUILD)
  plm_add_executable (dcmtk_test "${DCMTK_TEST_SRC}"
    "${PLASTIMATCH_LIBS}" "${GPUIT_EXE_LDFLAGS}" 
    ${BUILD_ALWAYS} ${INSTALL_NEVER})
endif ()



plm_add_executable (test_eps "${TEST_EPS_SRC}"
  "" "" 
  ${BUILD_IF_NOT_SLICER_EXT} ${INSTALL_IF_NOT_DEBIAN})
plm_add_executable (test_limits "${TEST_LIMITS_SRC}"
  "" "" 
  ${BUILD_IF_NOT_SLICER_EXT} ${INSTALL_IF_NOT_DEBIAN})
plm_add_executable (test_overflow "${TEST_OVERFLOW_SRC}"
  "" "" 
  ${BUILD_IF_NOT_SLICER_EXT} ${INSTALL_IF_NOT_DEBIAN})
