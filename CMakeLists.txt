##-----------------------------------------------------------------------------
##  Welcome to the Plastimatch CMakeLists.txt file
##-----------------------------------------------------------------------------
##  See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
##-----------------------------------------------------------------------------
CMAKE_MINIMUM_REQUIRED (VERSION 2.6)
PROJECT (plastimatch)

## Tell CMake to be quiet
IF(${CMAKE_MAJOR_VERSION}.${CMAKE_MINOR_VERSION} GREATER 2.4)
  CMAKE_POLICY (SET CMP0003 NEW)
ENDIF(${CMAKE_MAJOR_VERSION}.${CMAKE_MINOR_VERSION} GREATER 2.4)

##-----------------------------------------------------------------------------
##  Include files
##-----------------------------------------------------------------------------
INCLUDE (CheckFunctionExists)
INCLUDE (CheckTypeSize)

# http://www.cmake.org/pipermail/cmake/2008-December/025886.html
INCLUDE (InstallRequiredSystemLibraries)

# http://www.cmake.org/Bug/view.php?id=9220
INCLUDE (language_support.cmake)

##-----------------------------------------------------------------------------
##  Plastimatch configuration variables
##-----------------------------------------------------------------------------
SET (PLM_BUILD_TESTING ON CACHE BOOL "Enable regression testing")
SET (PLM_BUILD_CUDA_TEST OFF CACHE BOOL "Build Cuda test program")
SET (PLM_BUILD_OPENCL_TEST OFF CACHE BOOL "Build OpenCL test program")
SET (PLM_BUILD_OPENMP_TEST OFF CACHE BOOL "Build OpenMP test program")
SET (PLM_BUILD_QT_TEST OFF CACHE BOOL "Build Qt test program")
SET (PLM_BUILD_DEMONS_CUDA OFF CACHE BOOL "Build cuda version of demons")
SET (PLM_BUILD_SHARED_LIBS OFF CACHE BOOL "Build plastimatch as shared library")
SET (PLM_LINK_MSVCRT ON CACHE BOOL "Link against MSVCRT instead of LIBMCT")
SET (PLM_SLICER3_DIR ON CACHE PATH "Directory of Slicer3-Build")

##-----------------------------------------------------------------------------
##  Macros for creating targets
##-----------------------------------------------------------------------------
MACRO (PLM_ADD_LIBRARY 
    TARGET_NAME TARGET_SRC TARGET_LIBS)

  IF (BUILD_AGAINST_SLICER3)
    ADD_LIBRARY (${TARGET_NAME} STATIC ${TARGET_SRC})
    SLICER3_SET_PLUGINS_OUTPUT_PATH (${TARGET_NAME})
  ELSE (BUILD_AGAINST_SLICER3)
    ADD_LIBRARY (${TARGET_NAME} ${TARGET_SRC})
    INSTALL (TARGETS ${TARGET_NAME}
      RUNTIME DESTINATION bin
      LIBRARY DESTINATION lib
      ARCHIVE DESTINATION lib)
  ENDIF (BUILD_AGAINST_SLICER3)
  TARGET_LINK_LIBRARIES (${TARGET_NAME} ${TARGET_LIBS})
ENDMACRO (PLM_ADD_LIBRARY)
    
MACRO (PLM_ADD_EXECUTABLE 
    TARGET_NAME TARGET_SRC TARGET_LIBS TARGET_LDFLAGS)

  IF (NOT BUILD_AGAINST_SLICER3)
    ADD_EXECUTABLE(${TARGET_NAME} ${TARGET_SRC})
    TARGET_LINK_LIBRARIES(${TARGET_NAME} ${TARGET_LIBS})

    IF(NOT ${TARGET_LDFLAGS} STREQUAL "")
      SET_TARGET_PROPERTIES(${TARGET_NAME} 
	PROPERTIES LINK_FLAGS ${TARGET_LDFLAGS})
    ENDIF(NOT ${TARGET_LDFLAGS} STREQUAL "")
    INSTALL(TARGETS ${TARGET_NAME} DESTINATION bin)
  ENDIF (NOT BUILD_AGAINST_SLICER3)
ENDMACRO (PLM_ADD_EXECUTABLE)


MACRO (PLM_ADD_SLICER_EXECUTABLE 
    TARGET_NAME TARGET_SRC TARGET_LIBS TARGET_LDFLAGS)

  GENERATECLP (${TARGET_SRC} ${TARGET_NAME}.xml)

  ADD_EXECUTABLE (${TARGET_NAME} ${TARGET_SRC})
  TARGET_LINK_LIBRARIES (${TARGET_NAME} ${TARGET_LIBS})

  IF (NOT ${TARGET_LDFLAGS} STREQUAL "")
    SET_TARGET_PROPERTIES (${TARGET_NAME} 
      PROPERTIES LINK_FLAGS ${TARGET_LDFLAGS})
  ENDIF (NOT ${TARGET_LDFLAGS} STREQUAL "")

  SLICER3_SET_PLUGINS_OUTPUT_PATH (${TARGET_NAME})
  SLICER3_INSTALL_PLUGINS (${TARGET_NAME})

ENDMACRO (PLM_ADD_SLICER_EXECUTABLE)

##-----------------------------------------------------------------------------
##  Search for languages and language features
##-----------------------------------------------------------------------------

# C/C++ 32/64
CHECK_TYPE_SIZE("unsigned int"    CMAKE_SIZEOF_UINT)
CHECK_TYPE_SIZE("unsigned long"   CMAKE_SIZEOF_ULONG)

# Fortran
IF (NOT MINGW)
  # For some reason, CMake tries to invoke devenv instead of gfortran
  # for mingw.
  INCLUDE (PLM_FindFortran.cmake)

  # The FortranCInterface is not included on CMake 2.6.2
  # INCLUDE (FortranCInterface)
ENDIF (NOT MINGW)
INCLUDE (PLM_FindF2C.cmake)

# OpenMP
INCLUDE (PLM_FindOpenMP.cmake)

##-----------------------------------------------------------------------------
##  Search for include files
##-----------------------------------------------------------------------------
CHECK_INCLUDE_FILE ("sys/stat.h" HAVE_SYS_STAT)

##-----------------------------------------------------------------------------
##  Search for libraries
##-----------------------------------------------------------------------------
INCLUDE (FindOpenGL)
FIND_PACKAGE (wxWidgets)
FIND_PACKAGE (Qt4 COMPONENTS QtCore QtGui OPTIONAL)
IF (QT4_FOUND)
  MESSAGE (STATUS "Looking for Qt4 - found")
  INCLUDE (${QT_USE_FILE})
ELSE (QT4_FOUND)
  MESSAGE (STATUS "Looking for Qt4 - not found")
ENDIF (QT4_FOUND)

INCLUDE (PLM_FindBrook.cmake)
INCLUDE (PLM_Etags.cmake)
INCLUDE (PLM_FindDCMTK.cmake)
INCLUDE (PLM_FindFFTW.cmake)
INCLUDE (PLM_FindSubversion.cmake)
#INCLUDE(FindPantheios.cmake)

SET (CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}" ${CMAKE_MODULE_PATH})
FIND_PACKAGE (OpenCL)

##-----------------------------------------------------------------------------
##  Special CUDA processing
##-----------------------------------------------------------------------------
IF (MINGW)
  # Cuda doesn't work with mingw at all
  SET (CUDA_FOUND FALSE)
ELSE (MINGW)
  # In cmake 2.8, use stock findcuda.  Otherwise, use included one.
  IF (${CMAKE_MAJOR_VERSION}.${CMAKE_MINOR_VERSION} LESS 2.8)
    SET (CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/FindCUDA/CMake/cuda" 
      ${CMAKE_MODULE_PATH})
  ENDIF (${CMAKE_MAJOR_VERSION}.${CMAKE_MINOR_VERSION} LESS 2.8)
  FIND_PACKAGE (CUDA QUIET)
ENDIF (MINGW)
SET (CUDA_FOUND ${CUDA_FOUND} CACHE BOOL "Did we find cuda?")

IF (CUDA_FOUND AND NOT CUDA_CUT_INCLUDE_DIR)
  SET (CUDA_FOUND FALSE)
ENDIF (CUDA_FOUND AND NOT CUDA_CUT_INCLUDE_DIR)
IF(CUDA_FOUND)
  #  SET (CUDA_ATTACH_VS_BUILD_RULE_TO_CUDA_FILE OFF)
  CUDA_INCLUDE_DIRECTORIES (
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CUDA_CUT_INCLUDE_DIR}
    )
ENDIF(CUDA_FOUND)

##-----------------------------------------------------------------------------
##  Set the math library
##-----------------------------------------------------------------------------
IF(UNIX)
  SET (MATH_LIB -lm)
ELSE(UNIX)
  SET (MATH_LIB)
ENDIF(UNIX)

##-----------------------------------------------------------------------------
##  User feedback on which languages and libraries were found
##-----------------------------------------------------------------------------
IF(wxWidgets_FOUND)
  MESSAGE (STATUS "Looking for wxWidgets - found.")
ELSE(wxWidgets_FOUND)
  MESSAGE (STATUS "Looking for wxWidgets - not found.")
ENDIF(wxWidgets_FOUND)
IF(DCMTK_FOUND)
  MESSAGE (STATUS "Looking for dcmtk - found.")
ELSE(DCMTK_FOUND)
  MESSAGE (STATUS "Looking for dcmtk - not found.")
ENDIF(DCMTK_FOUND)

##-----------------------------------------------------------------------------
##  Slicer front-ends
##-----------------------------------------------------------------------------
find_package (Slicer3 QUIET)

#SET (Slicer3_DIR "" CACHE PATH "Build Slicer front-end for Plastimatch")
IF (Slicer3_DIR)
  MESSAGE (STATUS "Looking for Slicer3 - found " ${Slicer3_DIR})
  find_package (Slicer3 REQUIRED)
  include (${Slicer3_USE_FILE})

  # Set reasonable default install prefix and output paths
  # (after setting Slicer3_DIR, delete CMAKE_INSTALL_PREFIX and re-configure)
  slicer3_set_default_install_prefix_for_external_projects ()
  #  IF (BUILD_AGAINST_SLICER3)
  #  ENDIF (BUILD_AGAINST_SLICER3)

  #slicer3_enable_plugins_testing()

ELSE (Slicer3_DIR)
  MESSAGE (STATUS "Looking for Slicer3 - not found")
ENDIF (Slicer3_DIR)

# Slicer seems to require a shared library build
IF (Slicer3_DIR)
  SET (PLM_BUILD_SHARED_LIBS ON)
ENDIF (Slicer3_DIR)


##-----------------------------------------------------------------------------
##  I can't remember why this is necessary.  But at one time I needed 
##  it for compiling on AMD linux 64-bit.
##-----------------------------------------------------------------------------
IF("${CMAKE_SYSTEM_PROCESSOR}" STREQUAL "x86_64")
  SET (CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -fPIC")
  SET (CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -fPIC")
  SET (CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -fPIC")
  SET (CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -fPIC")
ENDIF("${CMAKE_SYSTEM_PROCESSOR}" STREQUAL "x86_64")

##-----------------------------------------------------------------------------
##  Set default build type to "Release"
##-----------------------------------------------------------------------------
##  Unfortunately the below doesn't work.  It gets
##  reset by the ITK include.
##-----------------------------------------------------------------------------
#IF(NOT CMAKE_BUILD_TYPE)
#  SET (CMAKE_BUILD_TYPE Release CACHE STRING
#      "Choose the type of build, options are: None Debug Release RelWithDebInfo MinSizeRel."	
#      FORCE)
#ENDIF(NOT CMAKE_BUILD_TYPE)

##-----------------------------------------------------------------------------
##  Get plastimatch version from subversion 
##-----------------------------------------------------------------------------
SET (PLASTIMATCH_SVN_VERSION "EXPERIMENTAL")
IF(EXISTS "${CMAKE_SOURCE_DIR}/.svn")
  IF(Subversion_FOUND)
    SET (PLASTIMATCH_EXPERIMENTAL TRUE)
    #Subversion_WC_INFO(${CMAKE_SOURCE_DIR} "svn")
    EXECUTE_PROCESS(
      COMMAND ${Subversion_SVNVERSION_EXECUTABLE} "--no-newline" ${CMAKE_SOURCE_DIR}
      RESULT_VARIABLE svn_result
      OUTPUT_VARIABLE svn_version
      ERROR_VARIABLE svn_version
      )
    IF(${svn_result} EQUAL 0)
      MESSAGE(STATUS "Looking for version in svn - found")
      SET (PLASTIMATCH_SVN_VERSION ${svn_version})
    ELSE(${svn_result} EQUAL 0)
      MESSAGE(STATUS "Looking for version in svn - can't connect")
    ENDIF(${svn_result} EQUAL 0)
  ELSE(Subversion_FOUND)
    MESSAGE(STATUS "Looking for version in svn - svn not found")
  ENDIF(Subversion_FOUND)
ELSE(EXISTS "${CMAKE_SOURCE_DIR}/.svn")
  SET (PLASTIMATCH_EXPERIMENTAL FALSE)
ENDIF(EXISTS "${CMAKE_SOURCE_DIR}/.svn")

##-----------------------------------------------------------------------------
##  ITK LIBRARY
##-----------------------------------------------------------------------------
FIND_PACKAGE(ITK)
IF(ITK_FOUND)
  INCLUDE(${ITK_USE_FILE})
  SET (ITK_VERSION 
    "${ITK_VERSION_MAJOR}.${ITK_VERSION_MINOR}.${ITK_VERSION_PATCH}")
ENDIF(ITK_FOUND)

##-----------------------------------------------------------------------------
##  WXWINDOWS LIBRARY
##-----------------------------------------------------------------------------
IF(wxWidgets_FOUND)
  IF(wxWidgets_CXX_FLAGS)
    string(REGEX REPLACE "-D" "" wxWidgets_DEFINITIONS 
      "${wxWidgets_DEFINITIONS}")
    SET_SOURCE_FILES_PROPERTIES(mondoshot_main.cpp
      PROPERTIES COMPILE_FLAGS 
      ${wxWidgets_CXX_FLAGS}
      )
  ENDIF(wxWidgets_CXX_FLAGS)
ENDIF(wxWidgets_FOUND)

##-----------------------------------------------------------------------------
##  DCMTK LIBRARY
##-----------------------------------------------------------------------------
SET (DCMTK_HAVE_CONFIG_H FALSE)
IF (UNIX)
  SET (DCMTK_HAVE_CONFIG_H TRUE)
ENDIF (UNIX)

##-----------------------------------------------------------------------------
##  Figure out library dependencies
##-----------------------------------------------------------------------------
SET (GPUIT_LIBRARY_DEPENDENCIES "")

IF (BROOK_FOUND)
  SET (GPUIT_LIBRARY_DEPENDENCIES
    ${GPUIT_LIBRARY_DEPENDENCIES}
    ${BROOK_LIBRARIES}
    ${OPENGL_LIBRARIES}
    ${DIRECTX_LIBRARIES}
    )
ENDIF (BROOK_FOUND)

IF(CUDA_FOUND)
  SET (GPUIT_LIBRARY_DEPENDENCIES
    ${GPUIT_LIBRARY_DEPENDENCIES}
    ${CUDA_LIBRARIES}
    )
ENDIF(CUDA_FOUND)

IF (FFTW_FOUND)
  SET (GPUIT_LIBRARY_DEPENDENCIES 
    ${GPUIT_LIBRARY_DEPENDENCIES}
    ${FFTW_LIBRARIES})
ENDIF (FFTW_FOUND)

IF (CMAKE_Fortran_COMPILER_WORKS)
  # It would seem that mixed-language programs need to set compiler and 
  # linker flags manually
  #   http://www.vtk.org/Wiki/CMakeForFortranExample
  # In theory, this is fixed in a future release
  #   http://www.cmake.org/Bug/view.php?id=9195
  MESSAGE (STATUS "Fortran compiler = ${CMAKE_Fortran_COMPILER}")
  IF (Fortran_COMPILER_NAME STREQUAL "gfortran" 
      OR Fortran_COMPILER_NAME STREQUAL "f95")
    SET (GPUIT_LIBRARY_DEPENDENCIES
      ${GPUIT_LIBRARY_DEPENDENCIES}
      gfortran
      )
  ENDIF (Fortran_COMPILER_NAME STREQUAL "gfortran" 
    OR Fortran_COMPILER_NAME STREQUAL "f95")
ENDIF (CMAKE_Fortran_COMPILER_WORKS)

IF(NOT CMAKE_Fortran_COMPILER_WORKS AND F2C_LIBRARY)
  SET (GPUIT_LIBRARY_DEPENDENCIES
    ${GPUIT_LIBRARY_DEPENDENCIES}
    ${F2C_LIBRARY}
    )
  IF(UNIX)
    SET (GPUIT_LIBRARY_DEPENDENCIES
      ${GPUIT_LIBRARY_DEPENDENCIES}
      f2c_helper
      )
  ENDIF(UNIX)
ENDIF(NOT CMAKE_Fortran_COMPILER_WORKS AND F2C_LIBRARY)

IF (OPENMP_FOUND)
  SET (GPUIT_LIBRARY_DEPENDENCIES
    ${GPUIT_LIBRARY_DEPENDENCIES}
    ${OPENMP_LIBRARIES}
    )
ENDIF (OPENMP_FOUND)

SET (GPUIT_LIBRARY_DEPENDENCIES
  ${GPUIT_LIBRARY_DEPENDENCIES}
  ${MATH_LIB}
  )

SET (GPUIT_LIBRARIES gpuit ${GPUIT_LIBRARY_DEPENDENCIES})

SET (PLASTIMATCH_LIBS plastimatch1 ${ITK_LIBRARIES} ${GPUIT_LIBRARIES})

SET (SLICER_PLUGIN_LIBS ${ITK_LIBRARIES} ${GPUIT_LIBRARY_DEPENDENCIES})

##-----------------------------------------------------------------------------
##  BROOK FILES
##-----------------------------------------------------------------------------
IF (BROOK_FOUND)
  BRCC_FILE(fdk_brook_kernel.br)
  SET_SOURCE_FILES_PROPERTIES(fdk_brook.cpp PROPERTIES
    OBJECT_DEPENDS "fdk_brook_kernel.cpp")

  BRCC_FILE(convolution_brook_kernel.br)
  BRCC_FILE(demons_brook_kernel.br)
  SET_SOURCE_FILES_PROPERTIES(demons_brook_kernel_wrap.cpp PROPERTIES
    OBJECT_DEPENDS "demons_brook_kernel.cpp")
  SET_SOURCE_FILES_PROPERTIES(convolution_brook_kernel_wrap.cpp PROPERTIES
    OBJECT_DEPENDS "convolution_brook_kernel.cpp")

  BRCC_FILE(toy_kernels.br)
  SET_SOURCE_FILES_PROPERTIES(toy_kernels.cpp PROPERTIES
    OBJECT_DEPENDS "toy_kernels.cpp")
ENDIF (BROOK_FOUND)

##-----------------------------------------------------------------------------
##  GPUIT/PLASTIMATCH SOURCE FILES
##-----------------------------------------------------------------------------
SET (GPUIT_LIBRARY_SRC
  bspline.c bspline.h
  bspline_opts.c bspline_opts.h
  bstrlib.c bstrlib.h
  demons.c demons.h
  demons_cpu.c
  demons_misc.c demons_misc.h
  file_util.c file_util.h
  fwrite_block.c fwrite_block.h
  logfile.c logfile.h
  mathutil.h
  print_and_exit.c print_and_exit.h 
  proj_image.c proj_image.h
  proj_matrix.c proj_matrix.h 
  readmha.c readmha.h
  timer.c timer.h
  vf.c vf.h
  volume.c volume.h
  )

IF(BROOK_FOUND)
  SET (GPUIT_LIBRARY_SRC
	${GPUIT_LIBRARY_SRC}
	demons_brook.cpp
	demons_brook_kernel_wrap.cpp
	demons_brook_kernel.br
	demons_brook_kernel.h
	convolution_brook_kernel.br
	convolution_brook_kernel_wrap.cpp
	convolution_brook_kernel.h
	)
ENDIF(BROOK_FOUND)

IF(CUDA_FOUND)
  SET (GPUIT_LIBRARY_SRC
    ${GPUIT_LIBRARY_SRC}
    bspline_cuda.cpp
    bspline_cuda.cu
    bspline_cuda.h
    )
  CUDA_COMPILE(BSPLINE_CUDA_WRAPPERS 
    bspline_cuda.cu)
  SET (GPUIT_LIBRARY_SRC
    ${GPUIT_LIBRARY_SRC}
    ${BSPLINE_CUDA_WRAPPERS}
    )
ENDIF(CUDA_FOUND)

IF (FFTW_FOUND)
  SET (GPUIT_LIBRARY_SRC
    ${GPUIT_LIBRARY_SRC}
    ramp_filter.c ramp_filter.h
   )
ENDIF (FFTW_FOUND)

IF (CMAKE_Fortran_COMPILER_WORKS OR F2C_LIBRARY)
  SET (GPUIT_LIBRARY_SRC
	${GPUIT_LIBRARY_SRC}
	bspline_optimize_lbfgsb.c bspline_optimize_lbfgsb.h
	)
ENDIF (CMAKE_Fortran_COMPILER_WORKS OR F2C_LIBRARY)
IF (CMAKE_Fortran_COMPILER_WORKS)
  SET (GPUIT_LIBRARY_SRC
	${GPUIT_LIBRARY_SRC}
	lbfgsb_2_1.f
	)
ENDIF (CMAKE_Fortran_COMPILER_WORKS)
IF (NOT CMAKE_Fortran_COMPILER_WORKS AND F2C_LIBRARY)
  SET (GPUIT_LIBRARY_SRC
	${GPUIT_LIBRARY_SRC}
	lbfgsb_2_1.c
	plm_f2c.h plm_f2c_win32.h
	)
ENDIF (NOT CMAKE_Fortran_COMPILER_WORKS AND F2C_LIBRARY)

IF(F2C_LIBRARY)
  SET (F2C_HELPER_LIBRARY_SRC
	main__.c
	)
ENDIF(F2C_LIBRARY)

SET (PLASTIMATCH_LIBRARY_SRC
	cxt_apply_dicom.cxx cxt_apply_dicom.h 
	cxt_io.cxx cxt_io.h 
	gcs_metric.h gcs_metric.hxx 
	gdcm_rtss.cxx gdcm_rtss.h
	gdcm_series.cxx gdcm_series.h
	gdcm_series_helper_2.cxx gdcm_series_helper_2.h
	gpuit_bspline.cxx gpuit_bspline.h 
	gpuit_demons.cxx gpuit_demons.h 
	itk_demons.cxx itk_demons.h 
	itk_dicom.cxx itk_dicom.h 
	itk_image.cxx itk_image.h 
	itk_image_cast.cxx itk_image_cast.h 
	itk_pointset.cxx itk_pointset.h
	itk_optim.cxx itk_optim.h 
	itk_registration.cxx itk_registration.h 
	itk_tps.cxx itk_tps.h
	itk_warp.cxx itk_warp.h 
	plm_image.cxx plm_image.h 
	plm_image_header.cxx plm_image_header.h 
	plm_image_type.cxx plm_image_type.h 
	plm_int.h
	plm_parms.cxx 
	plm_stages.cxx 
	plm_registration.h 
	plm_version.h 
	resample_mha.cxx resample_mha.h 
	slice_extract.cxx slice_extract.h
	synthetic_mha.cxx synthetic_mha.h
	tps.cxx tps.h
	xio_io.cxx xio_io.h
	xform.cxx xform.h 
	)

IF(ITK_USE_REVIEW)
  SET (PLASTIMATCH_LIBRARY_SRC
    ${PLASTIMATCH_LIBRARY_SRC}
    cxt_extract.cxx cxt_extract.h)
ENDIF(ITK_USE_REVIEW)

##-----------------------------------------------------------------------------
##  SOURCE FILES (EXECUTABLES)
##-----------------------------------------------------------------------------
SET (BSPLINE_SRC
	bspline_main.c 
	)
SET (CMS_DOSE_TO_MHA_SRC
	cms_dose_to_mha.c
	)
SET (COMPUTE_DISTANCE_SRC
	compute_distance.cxx compute_distance.h
	closest_point.cxx 
	read_obj.cxx
	)
SET (CUDA_PROBE_SRC
	cuda_probe_main.c
	cuda_probe.cu
	cuda_probe.h
	)
IF (CUDA_FOUND)
  CUDA_COMPILE(CUDA_PROBE_WRAPPERS cuda_probe.cu)
  SET (CUDA_PROBE_SRC
    ${CUDA_PROBE_SRC}
    ${CUDA_PROBE_WRAPPERS}
    )
ENDIF (CUDA_FOUND)
SET (CXT_APPLY_DICOM_SRC
  cxt_apply_dicom_main.cxx)
SET (CXT_TO_DICOM_RTSS_SRC 
  cxt_to_dicom_rtss.cxx)
SET (CXT_TO_MHA_SRC
	cxt_to_mha.cxx
	render_polyline.cxx
	render_polyline.h
	)
SET (DEMONS_SRC
	demons_main.c
	demons_opts.c demons_opts.h
	)
SET (DEMONS_CUDA_SRC
	readmha.c readmha.h
	volume.c volume.h
	)
SET (DICE_STATS_SRC
	contour_statistics.cxx contour_statistics.h
	dice_statistics.cxx 
	)
SET (DICOM_SERIES_TEST_SRC
  dicom_series_test.cxx)
SET (DICOM_RTSS_TO_CXT_SRC 
  dicom_rtss_to_cxt.cxx)
SET (DICOM_UID_SRC
	dicom_uid_main.cxx
	dicom_uid.cxx
	dicom_uid.h
	)
SET (DICOMRT_IMPORT_SLICER_SRC
        dicomrt-import-slicer.cxx
	)
SET (DRR_SRC
	drr_c.c drr.h
	drr_opts.c drr_opts.h
	drr_trilin.c drr_trilin.h
	)
IF (CUDA_FOUND)
  CUDA_COMPILE (CUDA_WRAPPERS drr_cuda.cu)
  SET (DRR_SRC
    ${DRR_SRC}
    ${CUDA_WRAPPERS}
    drr_cuda.cu drr_cuda.h
    fdk_utils.c fdk_utils.h    ## Temporary
    )
ENDIF (CUDA_FOUND)

SET (EXTRACT_CONTOUR_SRC
	extract_contour.cxx
	)
SET (FDK_SRC
        fdk_main.c fdk.h
	fdk_opts.c fdk_opts.h
	fdk_utils.c fdk_utils.h
	)
IF (CUDA_FOUND)
  SET (FDK_SRC
    ${FDK_SRC}
    fdk_cuda.cu fdk_cuda.h
    )
ENDIF (CUDA_FOUND)

IF (BROOK_FOUND)
  SET (FDK_SRC
    ${FDK_SRC}
    fdk_brook.cpp
    fdk_brook_kernel.br
    )
ENDIF (BROOK_FOUND)


SET (FDK_CUDA_INTEGRATED_NORMALIZED_SRC
	fdk_cuda.h fdk.h
	fdk_opts.c fdk_opts.h
	fdk_utils.c fdk_utils.h
	mathutil.h
	fdk_cuda_integrated_main.c
	fdk_cuda_ext.cu
	MGHMtx_opts.h
	ramp_filter.c ramp_filter.h 
	volume.c volume.h 
	writematrix_opts.c
	bowtie_correction.c
	)

IF (CUDA_FOUND)
  CUDA_COMPILE (FDK_CUDA_WRAPPERS fdk_cuda.cu)
  SET (FDK_SRC
    ${FDK_SRC}
    ${FDK_CUDA_WRAPPERS}
    )
  CUDA_COMPILE (FDK_CUDA_WRAPPERS fdk_cuda_ext.cu)
  SET (FDK_CUDA_INTEGRATED_NORMALIZED_SRC
    ${FDK_CUDA_INTEGRATED_NORMALIZED_SRC}
    ${FDK_CUDA_WRAPPERS}
    )
ENDIF (CUDA_FOUND)

SET (HND2RAW_ST_SRC
	hnd2raw_st.cpp
	)
SET (HND_TO_PFM_SRC
	hnd_to_pfm.c
	)
SET (ITK_TPS_WARP_SRC
        itk_tps_warp_main.cxx 
	)
SET (MASK_MHA_SRC
	mask_mha.cxx
	)
SET (MHA_TO_CMS_DOSE_SRC
	mha_to_cms_dose.c
	)
SET (MHA_TO_CXT_SRC
	mha_to_cxt.cxx
	)
SET (MHA_TO_RTOG_DOSE_SRC
	mha_to_rtog_dose.cxx
	)
SET (MONDOSHOT_SRC
	dicom_uid.cxx
	dicom_uid.h
	mondoshot_dicom.cpp
	mondoshot_dicom.h
	mondoshot_main.cpp
	mondoshot_main.h
        sqlite3.c
	sqlite3.h
	sqlite3ext.h
	)
SET (MONDOSHOT_RELAY_SRC
	mondoshot_relay_main.cpp
	)
SET (PATIENT_MASK_SRC
	patient_mask.cxx
	)
SET (PLASTIMATCH_SRC
	adjust_main.cxx adjust_main.h
 	file_type.cxx file_type.h
	compare_main.cxx compare_main.h
	plastimatch_main.cxx
	resample_main.cxx resample_main.h
	stats_main.cxx stats_main.h
 	warp_dij_main.cxx 
	warp_mha_main.cxx 
	warp_main.cxx warp_main.h
 	warp_pointset_main.cxx 
	)
SET (PLASTIMATCH_SLICER_SRC
        plastimatch-slicer.cxx
	)
SET (PLASTIMATCH_SLICER_TPS_SRC
        plastimatch-slicer-tps.cxx
	)
SET (COMPUTE_MEAN_IMAGE_SRC
	compute_mean_image_main.cxx
	)
SET (RTOG_TO_MHA_SRC
	rtog_to_mha.cxx
	render_polyline.cxx render_polyline.h
	exchkeys.cxx exchkeys.h
	)
SET (SHUFFLE_MHA_SRC
	shuffle_mha_main.cxx
	)
SET (SYNTHETIC_MHA_SRC
	synthetic_mha_main.cxx synthetic_mha_main.h
	)
SET (TPS_WARP_SRC
        tps_warp_main.cxx
        tps_warp_opts.cxx tps_warp_opts.h
	)
SET (VF_COMPARE_SRC
	vf_compare.c
	)
SET (VF_COMPOSE_SRC
	vf_compose.cxx
	)
SET (VF_INVERT_SRC
	vf_invert.cxx
	)
SET (VF_TO_VVF_SRC
	vf_to_vvf.cxx
	)
SET (VF_STATS_SRC
	vf_stats.c
	)
SET (XF_TO_XF_SRC
	xf_to_xf_main.cxx xf_to_xf_main.h
	)
SET (XIO_TO_CXT_SRC
	xio_to_cxt.cxx
	)

CHECK_FUNCTION_EXISTS(getopt_long HAVE_GETOPT_LONG)
IF(NOT HAVE_GETOPT_LONG)
    SET (CXT_APPLY_DICOM_SRC ${CXT_APPLY_DICOM_SRC} getopt_long.c getopt.h)
    SET (CXT_TO_DICOM_RTSS_SRC ${CXT_TO_DICOM_RTSS_SRC} getopt_long.c getopt.h)
    SET (CXT_TO_MHA_SRC ${CXT_TO_MHA_SRC} getopt_long.c getopt.h)
    SET (DICOM_RTSS_TO_CXT_SRC ${DICOM_RTSS_TO_CXT_SRC} getopt_long.c getopt.h)
    SET (DICOM_SERIES_TEST_SRC ${DICOM_SERIES_TEST_SRC} getopt_long.c getopt.h)
    SET (MHA_TO_CXT_SRC ${MHA_TO_CXT_SRC} getopt_long.c getopt.h)
    SET (MHA_TO_RTOG_DOSE_SRC ${MHA_TO_RTOG_DOSE_SRC} getopt_long.c getopt.h)
    SET (PATIENT_MASK_SRC ${PATIENT_MASK_SRC} getopt_long.c getopt.h)
    SET (PLASTIMATCH_SRC ${PLASTIMATCH_SRC} getopt_long.c getopt.h)
    SET (RTOG_TO_MHA_SRC ${RTOG_TO_MHA_SRC} getopt_long.c getopt.h)
    SET (COMPUTE_MEAN_IMAGE_SRC ${COMPUTE_MEAN_IMAGE_SRC} getopt_long.c getopt.h)
    SET (SYNTHETIC_MHA_SRC ${SYNTHETIC_MHA_SRC} getopt_long.c getopt.h)
    SET (VF_INVERT_SRC ${VF_INVERT_SRC} getopt_long.c getopt.h)
    SET (XIO_TO_CXT_SRC ${XIO_TO_CXT_SRC} getopt_long.c getopt.h)
    SET (XF_TO_XF_SRC ${XF_TO_XF_SRC} getopt_long.c getopt.h)
ENDIF(NOT HAVE_GETOPT_LONG)

##-----------------------------------------------------------------------------
##  SOURCE FILE PROPERTIES
##-----------------------------------------------------------------------------
IF(OPENMP_FOUND)
  SET_SOURCE_FILES_PROPERTIES(bspline.c
    PROPERTIES COMPILE_FLAGS ${OPENMP_FLAGS})
ENDIF(OPENMP_FOUND)

##-----------------------------------------------------------------------------
##  CONFIGURE INCLUDE FILE
##-----------------------------------------------------------------------------
CONFIGURE_FILE(${CMAKE_CURRENT_SOURCE_DIR}/plm_config.h.cmake 
  ${CMAKE_CURRENT_BINARY_DIR}/plm_config.h)

CONFIGURE_FILE(${CMAKE_CURRENT_SOURCE_DIR}/plm_f2c.h.cmake 
  ${CMAKE_CURRENT_BINARY_DIR}/plm_f2c.h)

IF (CMAKE_Fortran_COMPILER_WORKS)
##  Ideally I should do this if there is a real fortran compiler.
##  create_fortran_c_interface ("" setulb(int i) ${CMAKE_CURRENT_BINARY_DIR}/plm_f2c.h)
ENDIF (CMAKE_Fortran_COMPILER_WORKS)

##-----------------------------------------------------------------------------
##  INCLUDE DIRECTORIES
##-----------------------------------------------------------------------------
INCLUDE_DIRECTORIES (BEFORE ${CMAKE_CURRENT_SOURCE_DIR})
INCLUDE_DIRECTORIES (BEFORE ${CMAKE_CURRENT_BINARY_DIR})

IF(BROOK_FOUND)
  INCLUDE_DIRECTORIES (AFTER ${BROOK_INCLUDE_DIR})
ENDIF(BROOK_FOUND)

IF(CUDA_FOUND)
  INCLUDE_DIRECTORIES (AFTER ${CUDA_INCLUDE_DIRS} ${CUDA_CUT_INCLUDE_DIR})
ENDIF(CUDA_FOUND)

IF(DCMTK_FOUND)
  INCLUDE_DIRECTORIES (AFTER ${DCMTK_INCLUDE_DIR})
ENDIF(DCMTK_FOUND)

IF(PANTHEIOS_FOUND)
  INCLUDE_DIRECTORIES (AFTER ${STLSOFT_INCLUDE_DIR})
  INCLUDE_DIRECTORIES (AFTER ${PANTHEIOS_INCLUDE_DIR})
ENDIF(PANTHEIOS_FOUND)

IF(wxWidgets_FOUND)
  INCLUDE_DIRECTORIES (BEFORE ${wxWidgets_INCLUDE_DIRS})
ENDIF(wxWidgets_FOUND)

IF(FFTW_FOUND)
  INCLUDE_DIRECTORIES (BEFORE ${FFTW_INCLUDE_DIR})
  LINK_DIRECTORIES (${FFTWDIR})
ENDIF(FFTW_FOUND)

# IF (Qt4_FOUND)
#   INCLUDE_DIRECTORIES (AFTER ${Qt4_INCLUDE_DIRS})
# ENDIF (Qt4_FOUND)

##-----------------------------------------------------------------------------
##  Subdirectories
##-----------------------------------------------------------------------------
ADD_SUBDIRECTORY(fatm/src)
ADD_SUBDIRECTORY(CUDA_tex_test)

##-----------------------------------------------------------------------------
##  Library targets
##-----------------------------------------------------------------------------
IF (F2C_LIBRARY AND UNIX)
  PLM_ADD_LIBRARY (f2c_helper "${F2C_HELPER_LIBRARY_SRC}" "")
ENDIF (F2C_LIBRARY AND UNIX)

# IF (BUILD_AGAINST_SLICER3)
#   ADD_LIBRARY (gpuit STATIC ${GPUIT_LIBRARY_SRC})
#   SLICER3_SET_PLUGINS_OUTPUT_PATH (gpuit)
# ELSE (BUILD_AGAINST_SLICER3)
#   ADD_LIBRARY (gpuit ${GPUIT_LIBRARY_SRC})
#   TARGET_LINK_LIBRARIES (gpuit ${GPUIT_LIBRARY_DEPENDENCIES})
#   INSTALL (TARGETS gpuit
#     RUNTIME DESTINATION bin
#     LIBRARY DESTINATION lib
#     ARCHIVE DESTINATION lib)
# ENDIF (BUILD_AGAINST_SLICER3)

PLM_ADD_LIBRARY (gpuit "${GPUIT_LIBRARY_SRC}" "${GPUIT_LIBRARY_DEPENDENCIES}")

IF (ITK_FOUND)
  ## Despite claims to the contrary, you are not allowed to have a library
  ## with the same name as the executable.  What happens is the executable 
  ## tries to create a temporary library with the same name.  [MSVC 2005]
  ## That is why the library is called plastimatch1 instead of plastimatch.

#   IF (BUILD_AGAINST_SLICER3)
#     ADD_LIBRARY (plastimatch1 STATIC ${PLASTIMATCH_LIBRARY_SRC})
#     SLICER3_SET_PLUGINS_OUTPUT_PATH (plastimatch1)
#   ELSE (BUILD_AGAINST_SLICER3)
#     ADD_LIBRARY (plastimatch1 ${PLASTIMATCH_LIBRARY_SRC})
#     INSTALL (TARGETS plastimatch1 
#       RUNTIME DESTINATION bin
#       LIBRARY DESTINATION lib
#       ARCHIVE DESTINATION lib)
#   ENDIF (BUILD_AGAINST_SLICER3)
#   TARGET_LINK_LIBRARIES (plastimatch1 ${ITK_LIBRARIES} ${GPUIT_LIBRARIES})

  SET (PLASTIMATCH_LIBRARY_DEPENDENCIES ${ITK_LIBRARIES} ${GPUIT_LIBRARIES})
  PLM_ADD_LIBRARY (plastimatch1 "${PLASTIMATCH_LIBRARY_SRC}" 
    "${PLASTIMATCH_LIBRARY_DEPENDENCIES}")
ENDIF (ITK_FOUND)


##-----------------------------------------------------------------------------
##  Executable targets
##-----------------------------------------------------------------------------
PLM_ADD_EXECUTABLE (bspline "${BSPLINE_SRC}" "${GPUIT_LIBRARIES}" 
  "${OPENMP_LDFLAGS}")
PLM_ADD_EXECUTABLE (cms_dose_to_mha "${CMS_DOSE_TO_MHA_SRC}" "" "")
PLM_ADD_EXECUTABLE (demons "${DEMONS_SRC}" "${GPUIT_LIBRARIES}" 
  "${OPENMP_LDFLAGS}")
PLM_ADD_EXECUTABLE (drr "${DRR_SRC}" "${GPUIT_LIBRARIES}" 
  "${OPENMP_LDFLAGS}")
PLM_ADD_EXECUTABLE (fdk "${FDK_SRC}" "${GPUIT_LIBRARIES}" "${OPENMP_LDFLAGS}")
PLM_ADD_EXECUTABLE (hnd_to_pfm "${HND_TO_PFM_SRC}" "" "")
PLM_ADD_EXECUTABLE (hnd2raw_ST "${HND2RAW_ST_SRC}" "${GPUIT_LIBRARIES}" 
  "${OPENMP_LDFLAGS}")
PLM_ADD_EXECUTABLE (mha_to_raw mha_to_raw.c "" "")
PLM_ADD_EXECUTABLE (mha_to_cms_dose mha_to_cms_dose.c "" "")
PLM_ADD_EXECUTABLE (raw_to_mha raw_to_mha.c "" "")
PLM_ADD_EXECUTABLE (vf_compare "${VF_COMPARE_SRC}" "${GPUIT_LIBRARIES}" 
  "${OPENMP_LDFLAGS}")
PLM_ADD_EXECUTABLE (vf_stats "${VF_STATS_SRC}" "${GPUIT_LIBRARIES}" 
  "${OPENMP_LDFLAGS}")
PLM_ADD_EXECUTABLE (vf_to_vvf "${VF_TO_VVF_SRC}" "${GPUIT_LIBRARIES}" 
  "${OPENMP_LDFLAGS}")
PLM_ADD_EXECUTABLE (vf3d_to_mha vf3d_to_mha.c "" "")
PLM_ADD_EXECUTABLE (vox_to_mha vox_to_mha.c "" "")

IF(ITK_FOUND)
  PLM_ADD_EXECUTABLE (compute_distance "${COMPUTE_DISTANCE_SRC}" 
    "${PLASTIMATCH_LIBS}" "${OPENMP_LDFLAGS}")
  PLM_ADD_EXECUTABLE (cxt_apply_dicom "${CXT_APPLY_DICOM_SRC}" 
    "${PLASTIMATCH_LIBS}" "${OPENMP_LDFLAGS}")
  PLM_ADD_EXECUTABLE (cxt_to_dicom_rtss "${CXT_TO_DICOM_RTSS_SRC}" 
    "${PLASTIMATCH_LIBS}" "${OPENMP_LDFLAGS}")
  PLM_ADD_EXECUTABLE (cxt_to_mha "${CXT_TO_MHA_SRC}" 
    "${PLASTIMATCH_LIBS}" "${OPENMP_LDFLAGS}")
  IF(ITK_USE_REVIEW)
    PLM_ADD_EXECUTABLE (compute_mean_image "${COMPUTE_MEAN_IMAGE_SRC}"
      "${PLASTIMATCH_LIBS}" "${OPENMP_LDFLAGS}")
  ENDIF(ITK_USE_REVIEW)

  PLM_ADD_EXECUTABLE (dice_stats "${DICE_STATS_SRC}"
    "${PLASTIMATCH_LIBS}" "${OPENMP_LDFLAGS}")
  PLM_ADD_EXECUTABLE (dicom_to_mha dicom_to_mha.cxx
    "${PLASTIMATCH_LIBS}" "${OPENMP_LDFLAGS}")
  PLM_ADD_EXECUTABLE (dicom_rtss_to_cxt "${DICOM_RTSS_TO_CXT_SRC}"
    "${PLASTIMATCH_LIBS}" "${OPENMP_LDFLAGS}")
  PLM_ADD_EXECUTABLE (dicom_series_test "${DICOM_SERIES_TEST_SRC}"
    "${PLASTIMATCH_LIBS}" "${OPENMP_LDFLAGS}")
  IF (Slicer3_DIR)
    PLM_ADD_SLICER_EXECUTABLE (dicomrt-import-slicer 
      "${DICOMRT_IMPORT_SLICER_SRC}" "${PLASTIMATCH_LIBS}" "${OPENMP_LDFLAGS}")
  ENDIF (Slicer3_DIR)
  IF (ITK_USE_REVIEW)
    PLM_ADD_EXECUTABLE (extract_contour "${EXTRACT_CONTOUR_SRC}"
      "${PLASTIMATCH_LIBS}" "${OPENMP_LDFLAGS}")
  ENDIF (ITK_USE_REVIEW)
  PLM_ADD_EXECUTABLE (itk_tps_warp "${ITK_TPS_WARP_SRC}"
    "${PLASTIMATCH_LIBS}" "${OPENMP_LDFLAGS}")
  PLM_ADD_EXECUTABLE (mask_mha "${MASK_MHA_SRC}"
    "${PLASTIMATCH_LIBS}" "${OPENMP_LDFLAGS}")
  PLM_ADD_EXECUTABLE (merge_vfs merge_vector_fields.cxx
    "${PLASTIMATCH_LIBS}" "${OPENMP_LDFLAGS}")
  PLM_ADD_EXECUTABLE (merge2 merge2.cxx
    "${PLASTIMATCH_LIBS}" "${OPENMP_LDFLAGS}")
  IF (ITK_USE_REVIEW)
    PLM_ADD_EXECUTABLE (mha_to_cxt "${MHA_TO_CXT_SRC}"
      "${PLASTIMATCH_LIBS}" "${OPENMP_LDFLAGS}")
  ENDIF (ITK_USE_REVIEW)
  PLM_ADD_EXECUTABLE (mha_to_rtog_dose "${MHA_TO_RTOG_DOSE_SRC}"
    "${PLASTIMATCH_LIBS}" "${OPENMP_LDFLAGS}")
  PLM_ADD_EXECUTABLE (mha_to_vox mha_to_vox.c
    "${PLASTIMATCH_LIBS}" "${OPENMP_LDFLAGS}")
  PLM_ADD_EXECUTABLE (patient_mask "${PATIENT_MASK_SRC}"
    "${PLASTIMATCH_LIBS}" "${OPENMP_LDFLAGS}")
  PLM_ADD_EXECUTABLE (plastimatch "${PLASTIMATCH_SRC}"
    "${PLASTIMATCH_LIBS}" "${OPENMP_LDFLAGS}")
  IF (Slicer3_DIR)
    PLM_ADD_SLICER_EXECUTABLE (plastimatch-slicer 
      "${PLASTIMATCH_SLICER_SRC}" "${PLASTIMATCH_LIBS}" "${OPENMP_LDFLAGS}")
    PLM_ADD_SLICER_EXECUTABLE (plastimatch-slicer-tps 
      "${PLASTIMATCH_SLICER_TPS_SRC}" "${PLASTIMATCH_LIBS}" "${OPENMP_LDFLAGS}")
  ENDIF (Slicer3_DIR)
  PLM_ADD_EXECUTABLE (rtog_to_mha "${RTOG_TO_MHA_SRC}"
    "${PLASTIMATCH_LIBS}" "${OPENMP_LDFLAGS}")
  PLM_ADD_EXECUTABLE (shuffle_mha "${SHUFFLE_MHA_SRC}"
    "${PLASTIMATCH_LIBS}" "${OPENMP_LDFLAGS}")
  PLM_ADD_EXECUTABLE (synthetic_mha "${SYNTHETIC_MHA_SRC}"
    "${PLASTIMATCH_LIBS}" "${OPENMP_LDFLAGS}")
  PLM_ADD_EXECUTABLE (tps_warp "${TPS_WARP_SRC}" 
    "${PLASTIMATCH_LIBS}" "${OPENMP_LDFLAGS}")
  PLM_ADD_EXECUTABLE (union_mask union_mask.cxx
    "${PLASTIMATCH_LIBS}" "${OPENMP_LDFLAGS}")
  PLM_ADD_EXECUTABLE (vf_compose "${VF_COMPOSE_SRC}"
    "${PLASTIMATCH_LIBS}" "${OPENMP_LDFLAGS}")
  PLM_ADD_EXECUTABLE (vf_invert "${VF_INVERT_SRC}"
    "${PLASTIMATCH_LIBS}" "${OPENMP_LDFLAGS}")
  PLM_ADD_EXECUTABLE (xf_to_xf "${XF_TO_XF_SRC}"
    "${PLASTIMATCH_LIBS}" "${OPENMP_LDFLAGS}")
  PLM_ADD_EXECUTABLE (xio_to_cxt "${XIO_TO_CXT_SRC}" 
    "${PLASTIMATCH_LIBS}" "${OPENMP_LDFLAGS}")
ENDIF(ITK_FOUND)

# several executables require cuda
IF(CUDA_FOUND)
  IF (PLM_BUILD_DEMONS_CUDA)
    PLM_ADD_EXECUTABLE (demons_cuda "${DEMONS_CUDA_SRC}" 
      "${GPUIT_LIBRARIES}" "")
  ENDIF (PLM_BUILD_DEMONS_CUDA)
  IF (FFTW_FOUND)
    PLM_ADD_EXECUTABLE (fdk_cuda_integrated_normalized 
      "${FDK_CUDA_INTEGRATED_NORMALIZED_SRC}" 
      "${GPUIT_LIBRARIES}" 
      "${OPENMP_LDFLAGS}")
  ENDIF (FFTW_FOUND)
  PLM_ADD_EXECUTABLE (cuda_probe "${CUDA_PROBE_SRC}" "${CUDA_LIBRARIES}" "")
ENDIF(CUDA_FOUND)

# dicom_uid requires dcmtk
IF(DCMTK_FOUND)
  PLM_ADD_EXECUTABLE (dicom_uid "${DICOM_UID_SRC}" "${DCMTK_LIBRARIES}" "")
ENDIF(DCMTK_FOUND)

# mondoshot requires WIN32, wx, dcmtk
IF (WIN32 AND NOT CYGWIN AND wxWidgets_FOUND AND DCMTK_FOUND)
  ADD_EXECUTABLE(mondoshot WIN32 ${MONDOSHOT_SRC})
  TARGET_LINK_LIBRARIES(mondoshot ${wxWidgets_LIBRARIES} ${DCMTK_LIBRARIES})
  IF (NOT BUILD_AGAINST_SLICER3)
    INSTALL(TARGETS mondoshot DESTINATION bin)
  ENDIF (NOT BUILD_AGAINST_SLICER3)
ENDIF (WIN32 AND NOT CYGWIN AND wxWidgets_FOUND AND DCMTK_FOUND)

##-----------------------------------------------------------------------------
##  Additional install files
##-----------------------------------------------------------------------------
IF(ITK_FOUND)
  IF(EXISTS "${ITK_DIR}/bin/release/ITKCommon.dll")
    INSTALL(FILES "${ITK_DIR}/bin/release/ITKCommon.dll" DESTINATION bin)
  ENDIF(EXISTS "${ITK_DIR}/bin/release/ITKCommon.dll")

  INSTALL(FILES 
    "${CMAKE_SOURCE_DIR}/cttools/dicomrt_to_cxt.pl" 
    "${CMAKE_SOURCE_DIR}/cttools/dcm_image_uids.pl" 
    "${CMAKE_SOURCE_DIR}/cttools/cxt_apply_uids.pl"
    "${CMAKE_SOURCE_DIR}/cttools/parse_cxt.pl"
    DESTINATION bin)

  if (Slicer3_DIR)
    #slicer3_install_plugins(plastimatch1)
    #slicer3_install_plugins(gpuit)
  endif (Slicer3_DIR)
ENDIF(ITK_FOUND)

IF (CUDA_FOUND)
  IF (EXISTS "${CUDA_TOOLKIT_ROOT_DIR}/bin/cudart.dll")
    IF (NOT BUILD_AGAINST_SLICER3)
      INSTALL (FILES "${CUDA_TOOLKIT_ROOT_DIR}/bin/cudart.dll" DESTINATION bin)
    ENDIF (NOT BUILD_AGAINST_SLICER3)
  ENDIF (EXISTS "${CUDA_TOOLKIT_ROOT_DIR}/bin/cudart.dll")
ENDIF (CUDA_FOUND)

IF (NOT BUILD_AGAINST_SLICER3)
  INSTALL (FILES 
    COPYRIGHT.TXT
    INSTALL.TXT
    LICENSE.TXT
    README.TXT
    README.BROOK.TXT
    REGISTRATION.TXT
    DESTINATION doc)
ENDIF (NOT BUILD_AGAINST_SLICER3)

##-----------------------------------------------------------------------------
##  Simple test files
##-----------------------------------------------------------------------------
# Test executable -- cuda
IF (PLM_BUILD_CUDA_TEST AND CUDA_FOUND)
  CUDA_COMPILE (CUDA_TEST_WRAPPERS cuda_test.cu)
  SET (CUDA_TEST_SRC ${CUDA_TEST_WRAPPERS} cuda_test.cu)
  PLM_ADD_EXECUTABLE (cuda_test "${CUDA_TEST_SRC}" "${CUDA_LIBRARIES}" "")
  SET_TARGET_PROPERTIES (cuda_test PROPERTIES LINKER_LANGUAGE CXX)
ENDIF (PLM_BUILD_CUDA_TEST AND CUDA_FOUND)

# Test executable -- opencl
IF (PLM_BUILD_OPENCL_TEST AND OPENCL_FOUND)
  INCLUDE_DIRECTORIES (BEFORE ${OPENCL_INCLUDE_DIR})
  SET (OPENCL_TEST_SRC opencl_test.cxx opencl_test.h opencl_test.cl)
  PLM_ADD_EXECUTABLE (opencl_test "${OPENCL_TEST_SRC}" "${OPENCL_LIBRARIES}" "")
  #SET_TARGET_PROPERTIES (opencl_test PROPERTIES LINKER_LANGUAGE CXX)
ENDIF (PLM_BUILD_OPENCL_TEST AND OPENCL_FOUND)

# Test executable -- openmp
IF (PLM_BUILD_OPENMP_TEST AND OPENMP_FOUND)
  ADD_EXECUTABLE(openmp_test openmp_test.c)
  TARGET_LINK_LIBRARIES(openmp_test ${MATH_LIB} ${OPENMP_LIBRARIES})
  SET_TARGET_PROPERTIES(openmp_test PROPERTIES COMPILE_FLAGS ${OPENMP_FLAGS})
  MESSAGE(STATUS "OPENMP_LDFLAGS = ${OPENMP_LDFLAGS}")
  IF(NOT ${OPENMP_LDFLAGS} STREQUAL "")
    SET_TARGET_PROPERTIES(openmp_test PROPERTIES LINK_FLAGS 
      ${OPENMP_LDFLAGS})
  ENDIF(NOT ${OPENMP_LDFLAGS} STREQUAL "")
ENDIF (PLM_BUILD_OPENMP_TEST AND OPENMP_FOUND)

# Test executable -- qt
IF (PLM_BUILD_QT_TEST AND QT4_FOUND)
  SET (QT_TEST_SRC qt_test.cxx)
  PLM_ADD_EXECUTABLE (qt_test "${QT_TEST_SRC}" "${QT_LIBRARIES}" "")
ENDIF (PLM_BUILD_QT_TEST AND QT4_FOUND)

##-----------------------------------------------------------------------------
##  Regression testing
##  Rationale:
##    1) Tests need to be launched from a script, because 
##       CTest is not able to add the ITK DLL directory 
##       to the path.
##    2) Prefer use of CMAKE as scripting language 
##       to avoid requirement of outside package (e.g. python)
##  Notes/Issues:
##    "Release" is hard coded.  There is (apparently) 
##    no easy way to avoid this (i.e. no good suggestions 
##    on CMake/CTest email list.
##-----------------------------------------------------------------------------
SET (PLM_TESTING_SOURCE_DIR "${CMAKE_SOURCE_DIR}/Testing")
SET (PLM_TESTING_DATA_DIR "${CMAKE_SOURCE_DIR}/Testing/Data")
SET (PLM_TESTING_BUILD_DIR "${CMAKE_BINARY_DIR}/Testing")

# Figure out which tests to ignore
# Note: we need two copies, because unix users might run "make test" 
# in either directory
CONFIGURE_FILE(${PLM_TESTING_SOURCE_DIR}/CTestCustom.cmake
  ${CMAKE_BINARY_DIR}/CTestCustom.cmake
  @ONLY)
CONFIGURE_FILE(${PLM_TESTING_SOURCE_DIR}/CTestCustom.cmake
  ${CMAKE_BINARY_DIR}/Testing/CTestCustom.cmake
  @ONLY)

# Set the "hack" macros to find windows binaries (under Release subdir)
IF (WIN32 AND NOT CYGWIN AND NOT MINGW)
  SET (PLM_ITK_LIBRARY_PATH_HACK -DITK_LIBRARY_PATH=${ITK_LIBRARY_DIRS}/Release)
  SET (PLM_PLASTIMATCH_PATH_HACK ${CMAKE_CURRENT_BINARY_DIR}/Release)
  SET (PLM_PLASTIMATCH_TESTING_PATH_HACK 
    ${CMAKE_CURRENT_BINARY_DIR}/Testing/Release)
ELSE (WIN32 AND NOT CYGWIN AND NOT MINGW)
  SET (PLM_ITK_LIBRARY_PATH_HACK "")
  SET (PLM_PLASTIMATCH_PATH_HACK ${CMAKE_CURRENT_BINARY_DIR})
  SET (PLM_PLASTIMATCH_TESTING_PATH_HACK ${CMAKE_CURRENT_BINARY_DIR}/Testing)
ENDIF (WIN32 AND NOT CYGWIN AND NOT MINGW)

MACRO(PLM_ADD_TEST TESTNAME WORKING_DIR PLM_TEST_COMMAND PARMS)
  ADD_TEST (${TESTNAME} ${CMAKE_COMMAND} 
    -DTESTNAME=${TESTNAME}
    -DWORKING_DIR=${WORKING_DIR}
    ${PLM_ITK_LIBRARY_PATH_HACK}
    -DPLM_TEST_COMMAND=${PLM_TEST_COMMAND}
    -DPLM_TESTING_SOURCE_DIR=${PLM_TESTING_SOURCE_DIR}
    -DPLM_TESTING_BUILD_DIR=${PLM_TESTING_BUILD_DIR}
    "-DPARMS=${PARMS}"
    -P ${CMAKE_SOURCE_DIR}/RUN_CTEST.cmake
    )
ENDMACRO(PLM_ADD_TEST)

IF(PLM_BUILD_TESTING)
  ENABLE_TESTING()
  ADD_SUBDIRECTORY(Testing)
ENDIF(PLM_BUILD_TESTING)

##-----------------------------------------------------------------------------
##  CPACK stuff
##  Note: "make package_source" to create tarball
##-----------------------------------------------------------------------------
SET (CPACK_GENERATOR "ZIP")
SET (CPACK_SOURCE_GENERATOR "TBZ2")
SET (CPACK_PACKAGE_VERSION_MAJOR "1")
SET (CPACK_PACKAGE_VERSION_MAJOR "1")
SET (CPACK_PACKAGE_VERSION_MINOR "4-beta")
SET (CPACK_PACKAGE_VERSION_PATCH ${PLASTIMATCH_SVN_VERSION})
SET (CPACK_SOURCE_IGNORE_FILES
  "/.svn"
  "~$"
)
INCLUDE(CPack)
