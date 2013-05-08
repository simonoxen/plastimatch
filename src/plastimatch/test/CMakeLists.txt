##-----------------------------------------------------------------------------
##  See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
##-----------------------------------------------------------------------------
project (src_plastimatch_test)

include_directories (BEFORE ${CMAKE_CURRENT_SOURCE_DIR})

##-----------------------------------------------------------------------------
##  CMAKE FLAGS
##-----------------------------------------------------------------------------
option (PLM_BUILD_API_TEST    "Build Plastimatch API test programs" OFF)
option (PLM_BUILD_CPROP_TEST  "Build C++ properties test programs"  OFF)
option (PLM_BUILD_CUDA_TEST   "Build CUDA test program"   OFF)
option (PLM_BUILD_DCMTK_TEST  "Build DCMTK test program"  OFF)
option (PLM_BUILD_DLIB_TEST   "Build dlib test program"   OFF)
option (PLM_BUILD_GABOR_TEST  "Build Gabor test program"  OFF)
option (PLM_BUILD_GDCM1_TEST  "Build GDCM test program"   OFF)
option (PLM_BUILD_ITK_TEST    "Build ITK test program"    OFF)
option (PLM_BUILD_MEX_TEST    "Build mex test program"    OFF)
option (PLM_BUILD_NLOPT_TEST  "Build NLopt test program"  OFF)
option (PLM_BUILD_NM_TEST     "Build Nelder-Meade test program" OFF)
option (PLM_BUILD_OPENCL_TEST "Build OpenCL test program" OFF)
option (PLM_BUILD_OPENMP_TEST "Build OpenMP test program" OFF)
option (PLM_BUILD_SMARTP_TEST "Build Smart pointer test program" OFF)
option (PLM_BUILD_QT_TEST     "Build Qt test program"     OFF)
option (PLM_BUILD_RANSAC_TEST "Build RANSAC test program" OFF)

##-----------------------------------------------------------------------------
##  BUILD TARGETS
##-----------------------------------------------------------------------------
# Test executable -- plastimatch api
if (PLM_BUILD_API_TEST)
  plm_add_executable (api_test api_test.cxx
    "${PLASTIMATCH_LIBS}" "${PLASTIMATCH_LDFLAGS}" 
    ${BUILD_ALWAYS} ${INSTALL_NEVER})
endif ()

# Test executable -- c++ properties
if (PLM_BUILD_CPROP_TEST)
  plm_add_executable (test_eps test_eps.cxx
    "" "" 
    ${BUILD_IF_NOT_SLICER_EXT} ${INSTALL_NEVER})
  plm_add_executable (test_limits test_limits.cxx
    "" "" 
    ${BUILD_IF_NOT_SLICER_EXT} ${INSTALL_NEVER})
  plm_add_executable (test_overflow test_overflow.cxx
    "" "" 
    ${BUILD_IF_NOT_SLICER_EXT} ${INSTALL_NEVER})
endif ()

# Test executable -- cuda
if (CUDA_FOUND AND PLM_BUILD_CUDA_TEST)
  cuda_compile (CUDA_TEST_WRAPPERS cuda_test.cu)
  plm_add_executable (cuda_test "${CUDA_TEST_WRAPPERS}" 
    "${CUDA_LIBRARIES}" "" 
    ${BUILD_ALWAYS} ${INSTALL_NEVER})
  set_target_properties (cuda_test PROPERTIES LINKER_LANGUAGE CXX)

  # Test executable -- cuda_tex_test
  add_subdirectory (CUDA_tex_test)
endif ()

# Test executable -- dcmtk
if (ITK_FOUND AND DCMTK_FOUND AND PLM_BUILD_DCMTK_TEST)
  plm_add_executable (dcmtk_test dcmtk_test.cxx
    "${PLASTIMATCH_LIBS}" "${PLASTIMATCH_LDFLAGS}" 
    ${BUILD_ALWAYS} ${INSTALL_NEVER})
endif ()

# Test executable -- dlib
if (PLM_BUILD_DLIB_TEST)
  plm_add_executable (dlib_test dlib_test.cxx
    "${PLASTIMATCH_LIBS}" "${PLASTIMATCH_LDFLAGS}" 
    ${BUILD_ALWAYS} ${INSTALL_NEVER})
endif ()

# Test executable -- gabor_test
if (PLM_BUILD_GABOR_TEST)
  plm_add_executable (gabor_test gabor_test.cxx 
    "${PLASTIMATCH_LIBS}" "${GPUIT_EXE_LDFLAGS}" 
    ${BUILD_IF_NOT_SLICER_EXT} ${INSTALL_NEVER})
endif ()

# Test executable -- gdcm
if (ITK_FOUND AND GDCM_VERSION_1 AND PLM_BUILD_GDCM1_TEST)
  plm_add_executable (gdcm1_test gdcm1_test.cxx
    "${PLASTIMATCH_LIBS}" "${GPUIT_EXE_LDFLAGS}" 
    ${BUILD_ALWAYS} ${INSTALL_NEVER})
endif ()

# Test executable -- itk
if (ITK_FOUND AND PLM_BUILD_ITK_TEST)
  plm_add_executable (itk_test itk_test.cxx
    "${PLASTIMATCH_LIBS}" "${GPUIT_EXE_LDFLAGS}" 
    ${BUILD_ALWAYS} ${INSTALL_NEVER})
endif ()

# Test executable -- mex
if (PLM_BUILD_MEX_TEST)
  if (OCTAVE_FOUND)
    add_custom_command (
      OUTPUT "${CMAKE_BINARY_DIR}/mex_test.mex"
      COMMAND ${OCTAVE_MKOCTFILE} --mex 
      -o "${CMAKE_BINARY_DIR}/mex_test.mex" 
      "${CMAKE_SOURCE_DIR}/mex_test.c"
      DEPENDS "${CMAKE_SOURCE_DIR}/mex_test.c")
    add_custom_target (oct_mex_test
      DEPENDS "${CMAKE_BINARY_DIR}/mex_test.mex")
  endif ()
  if (MATLAB_FOUND)
    file (WRITE "${CMAKE_BINARY_DIR}/compile_mex_test.m"
      "mex \"${CMAKE_SOURCE_DIR}/mex_test.c\";exit;\n")
    file (WRITE "${CMAKE_BINARY_DIR}/mex_test.cmake"
      "EXECUTE_PROCESS (COMMAND ${MATLAB_EXE} -nosplash -nodesktop -nojvm
       -r compile_mex_test
       RESULT_VARIABLE RESULT
       OUTPUT_VARIABLE STDOUT
       ERROR_VARIABLE STDERR)\n")
    add_custom_command (
      OUTPUT "${CMAKE_BINARY_DIR}/mex_test${MATLAB_MEXEXT}"
      COMMAND ${CMAKE_COMMAND} -P "${CMAKE_BINARY_DIR}/mex_test.cmake"
      DEPENDS "${CMAKE_SOURCE_DIR}/mex_test.c")
    add_custom_target (mat_mex_test
      DEPENDS "${CMAKE_BINARY_DIR}/mex_test${MATLAB_MEXEXT}")
    target_link_libraries (mex_test ${MATLAB_LIBRARIES})
  endif ()
endif ()

# Test executable -- nlopt
if (PLM_BUILD_NLOPT_TEST AND NLOPT_FOUND)
  set (NLOPT_TEST_LIBS ${GPUIT_LIBRARIES} ${NLOPT_LIBRARIES})
  plm_add_executable (nlopt_test nlopt_test.c
    "${NLOPT_TEST_LIBS}" "${OPENMP_LDFLAGS}" 
    ${BUILD_ALWAYS} ${INSTALL_NEVER})
  set_target_properties (nlopt_test PROPERTIES LINKER_LANGUAGE CXX)
endif ()

# Test executable -- nm
if (PLM_BUILD_NM_TEST)
  plm_add_executable (nelder_mead_test nelder_mead_test.cxx
    "${PLASTIMATCH_LIBS}" "${OPENMP_LDFLAGS}" 
    ${BUILD_ALWAYS} ${INSTALL_NEVER})
endif ()

# Test executable -- opencl
if (PLM_BUILD_OPENCL_TEST AND OPENCL_FOUND)
  set (OPENCL_TEST_SRC
    opencl_test.cxx opencl_test.h opencl_test.cl
    "${CMAKE_BINARY_DIR}/opencl_test.cl"
    )
  set (OPENCL_TEST_LIBS ${OPENCL_LIBRARIES} ${PLASTIMATCH_LIBS})
  plm_add_executable (opencl_test "${OPENCL_TEST_SRC}" 
    "${OPENCL_TEST_LIBS}" "" 
    ${BUILD_ALWAYS} ${INSTALL_NEVER})

  # I don't yet know how to bundle the .cl file within the executable.
  # Therefore, copy the .cl into binary directory.
  add_custom_command (
    OUTPUT "${CMAKE_BINARY_DIR}/opencl_test.cl" 
    COMMAND ${CMAKE_COMMAND} "-E" "copy" 
    "${CMAKE_CURRENT_SOURCE_DIR}/opencl_test.cl" 
    "${CMAKE_BINARY_DIR}/opencl_test.cl" 
    DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/opencl_test.cl")
endif ()

# Test executable -- openmp
if (PLM_BUILD_OPENMP_TEST AND OPENMP_FOUND)
  set_source_files_properties (openmp_test.cxx
    PROPERTIES COMPILE_FLAGS ${OPENMP_FLAGS})
  plm_add_executable (openmp_test openmp_test.cxx
    "${PLASTIMATCH_LIBS}" "${OPENMP_LDFLAGS}" 
    ${BUILD_ALWAYS} ${INSTALL_NEVER})
endif ()

# Test executable -- smart pointers
if (PLM_BUILD_SMARTP_TEST)
  ## This is a test, in case of deciding to use shared_ptr
  find_package (TR1)

  plm_add_executable (test_smartp test_smartp.cxx
    "${PLASTIMATCH_LIBS}" "${GPUIT_EXE_LDFLAGS}" 
    ${BUILD_ALWAYS} ${INSTALL_NEVER})
endif ()

# Test executable -- qt
if (PLM_BUILD_QT_TEST AND QT4_FOUND)
  plm_add_executable (qt_test qt_test.cxx
    "${QT_LIBRARIES}" "" 
    ${BUILD_ALWAYS} ${INSTALL_NEVER})
endif ()

# Test executable -- ransac
if (PLM_BUILD_RANSAC_TEST)
  plm_add_executable (ransac_test ransac_test.cxx
    "${PLASTIMATCH_LIBS}" "${GPUIT_EXE_LDFLAGS}" 
    ${BUILD_IF_NOT_SLICER_EXT} ${INSTALL_NEVER})
endif ()

