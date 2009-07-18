cmake_minimum_required(VERSION 2.6)
PROJECT(plastimatch)

######################################################
##  Include files
###################################################### 	 
INCLUDE(CheckFunctionExists)
INCLUDE(FindOpenGL)
##INCLUDE(FindPantheios.cmake)
#INCLUDE(FindwxWidgets)
FIND_PACKAGE(wxWidgets)

INCLUDE(PLM_Etags.cmake)
INCLUDE(PLM_FindSubversion.cmake)
INCLUDE(PLM_FindDCMTK.cmake)
INCLUDE(PLM_FindFFTW.cmake)
INCLUDE(GPUIT.cmake)

######################################################
##  Help the user out on unix
###################################################### 	 
IF(DCMTK_FOUND)
  MESSAGE (STATUS "Looking for dcmtk - found.")
ELSE(DCMTK_FOUND)
  MESSAGE (STATUS "Looking for dcmtk - not found.")
ENDIF(DCMTK_FOUND)

IF(wxWidgets_FOUND)
  MESSAGE (STATUS "Looking for wxWidgets - found.")
ELSE(wxWidgets_FOUND)
  MESSAGE (STATUS "Looking for wxWidgets - not found.")
ENDIF(wxWidgets_FOUND)

######################################################
##  Plastimatch configuration variables
###################################################### 	 
SET(PLM_BUILD_TESTING OFF CACHE BOOL "Enable regression testing")
SET(BUILD_OPENMP_TEST OFF CACHE BOOL "Build OpenMP test program")
SET(BUILD_CUDA_TEST OFF CACHE BOOL "Build Cuda test program")
SET(BUILD_BSPLINE_BROOK OFF CACHE BOOL "Build brook version of B-splines")
SET(BUILD_DEMONS_CUDA OFF CACHE BOOL "Build cuda version of demons")
SET(PLM_BSPLINE_CXX OFF CACHE BOOL "Use C++ version of bspline code")
SET(PLM_BUILD_SHARED_LIBS OFF CACHE BOOL "Build plastimatch as shared library")
SET(PLASTIMATCH_SVN_VERSION "EXPERIMENTAL")



######################################################
##  SLICER FRONT-ENDS
###################################################### 	 
SET(Slicer3_DIR "" CACHE PATH "Build Slicer front-end for Plastimatch")

IF(Slicer3_DIR)
  cmake_minimum_required(VERSION 2.6)

  MESSAGE (STATUS "Looking for Slicer3 - found " ${Slicer3_DIR})

  find_package(Slicer3 REQUIRED)
  include(${Slicer3_USE_FILE})

  # Set reasonable default install prefix and output paths
  # (after setting Slicer3_DIR, delete CMAKE_INSTALL_PREFIX and re-configure)
  slicer3_set_default_install_prefix_for_external_projects()

  # The Command Line Module
  set(CLP plastimatch-slicer)
  set(${CLP}_SOURCE ${CLP}.cxx)
  generateclp(${CLP}_SOURCE ${CLP}.xml)

  # Build it
  add_executable(${CLP} ${${CLP}_SOURCE})
  slicer3_set_plugins_output_path(${CLP})
  target_link_libraries(${CLP} plastimatch1 ${ITK_LIBRARIES})

  # Install it
  slicer3_install_plugins(${CLP})

  # The Command Line Module
  set(CLP dicomrt-import-slicer)
  set(${CLP}_SOURCE ${CLP}.cxx)
  generateclp(${CLP}_SOURCE ${CLP}.xml)

  # Build it
  add_executable(${CLP} ${${CLP}_SOURCE})
  slicer3_set_plugins_output_path(${CLP})
  target_link_libraries(${CLP} plastimatch1 ${ITK_LIBRARIES})

  # Install it
  slicer3_install_plugins(${CLP})

  # Test it
  #enable_testing()
  #include(CTest)
  #set(UPDATE_TYPE svn)

  # ? 	 
  #slicer3_add_plugins_test(${CLP}Test1 ${CLP} --help)

ELSE(Slicer3_DIR)
  MESSAGE (STATUS "Looking for Slicer3 - not found")
ENDIF(Slicer3_DIR)

## Tell CMake to be quiet
IF(${CMAKE_MAJOR_VERSION}.${CMAKE_MINOR_VERSION} GREATER 2.4)
   cmake_policy(SET CMP0003 NEW)
ENDIF(${CMAKE_MAJOR_VERSION}.${CMAKE_MINOR_VERSION} GREATER 2.4)

######################################################
##  By default, brook builts /MT.  We need to match this.
######################################################
# IF(NOT GPUIT_LINK_MSVCRT)
#   STRING(REGEX REPLACE "/MD" "/MT" 
# 	     CMAKE_CXX_FLAGS_DEBUG 
# 	     "${CMAKE_CXX_FLAGS_DEBUG}")
#   STRING(REGEX REPLACE "/MD" "/MT"
# 	     CMAKE_CXX_FLAGS_MINSIZEREL 
# 	     "${CMAKE_CXX_FLAGS_MINSIZEREL}")
#   STRING(REGEX REPLACE "/MD" "/MT"
# 	     CMAKE_CXX_FLAGS_RELEASE 
# 	     "${CMAKE_CXX_FLAGS_RELEASE}")
#   STRING(REGEX REPLACE "/MD" "/MT"
# 	     CMAKE_CXX_FLAGS_RELWITHDEBINFO 
# 	     "${CMAKE_CXX_FLAGS_RELWITHDEBINFO}")
#   STRING(REGEX REPLACE "/MD" "/MT" 
# 	     CMAKE_C_FLAGS_DEBUG 
# 	     "${CMAKE_C_FLAGS_DEBUG}")
#   STRING(REGEX REPLACE "/MD" "/MT"
# 	     CMAKE_C_FLAGS_MINSIZEREL 
# 	     "${CMAKE_C_FLAGS_MINSIZEREL}")
#   STRING(REGEX REPLACE "/MD" "/MT"
# 	     CMAKE_C_FLAGS_RELEASE 
# 	     "${CMAKE_C_FLAGS_RELEASE}")
#   STRING(REGEX REPLACE "/MD" "/MT"
# 	     CMAKE_C_FLAGS_RELWITHDEBINFO 
# 	     "${CMAKE_C_FLAGS_RELWITHDEBINFO}")
# ENDIF(NOT GPUIT_LINK_MSVCRT)

IF("${CMAKE_SYSTEM_PROCESSOR}" STREQUAL "x86_64")
  SET(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -fPIC")
  SET(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -fPIC")
  SET(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -fPIC")
  SET(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -fPIC")
ENDIF("${CMAKE_SYSTEM_PROCESSOR}" STREQUAL "x86_64")

######################################################
##  Set default build type to "Release"
######################################################
##  Unfortunately the below doesn't work.  It gets
##  reset by the ITK include.
######################################################
#IF(NOT CMAKE_BUILD_TYPE)
#  SET(CMAKE_BUILD_TYPE Release CACHE STRING
#      "Choose the type of build, options are: None Debug Release RelWithDebInfo MinSizeRel."	
#      FORCE)
#ENDIF(NOT CMAKE_BUILD_TYPE)

######################################################
##  Get plastimatch version from subversion 
######################################################
IF(EXISTS "${CMAKE_SOURCE_DIR}/.svn")
  IF(Subversion_FOUND)
    SET(PLASTIMATCH_EXPERIMENTAL TRUE)
    #Subversion_WC_INFO(${CMAKE_SOURCE_DIR} "svn")
    EXECUTE_PROCESS(
      COMMAND ${Subversion_SVNVERSION_EXECUTABLE} "--no-newline" ${CMAKE_SOURCE_DIR}
      RESULT_VARIABLE svn_result
      OUTPUT_VARIABLE svn_version
      ERROR_VARIABLE svn_version
      )
    IF(${svn_result} EQUAL 0)
      MESSAGE(STATUS "Looking for version in svn - found")
      SET(PLASTIMATCH_SVN_VERSION ${svn_version})
    ELSE(${svn_result} EQUAL 0)
      MESSAGE(STATUS "Looking for version in svn - can't connect")
    ENDIF(${svn_result} EQUAL 0)
  ELSE(Subversion_FOUND)
    MESSAGE(STATUS "Looking for version in svn - svn not found")
  ENDIF(Subversion_FOUND)
ELSE(EXISTS "${CMAKE_SOURCE_DIR}/.svn")
  SET(PLASTIMATCH_EXPERIMENTAL FALSE)
ENDIF(EXISTS "${CMAKE_SOURCE_DIR}/.svn")

######################################################
##  MATH LIBRARY
######################################################
IF(UNIX)
  SET(MATH_LIB -lm)
ELSE(UNIX)
  SET(MATH_LIB)
ENDIF(UNIX)

######################################################
##  GPUIT LIBRARY
######################################################
SET(GPUIT_LIBRARY_DEPENDENCIES "")
IF(BROOK_FOUND)
  SET(GPUIT_LIBRARY_DEPENDENCIES
    ${GPUIT_LIBRARY_DEPENDENCIES}
    ${BROOK_LIBRARIES}
    ${OPENGL_LIBRARIES}
    ${DIRECTX_LIBRARIES}
    )
ENDIF(BROOK_FOUND)
IF(F2C_LIBRARY)
  SET(GPUIT_LIBRARY_DEPENDENCIES
    ${GPUIT_LIBRARY_DEPENDENCIES}
    ${F2C_LIBRARY}
    )
  IF(UNIX)
    SET(GPUIT_LIBRARY_DEPENDENCIES
      ${GPUIT_LIBRARY_DEPENDENCIES}
      f2c_helper
      )
  ENDIF(UNIX)
ENDIF(F2C_LIBRARY)
IF(CUDA_FOUND)
  SET(GPUIT_LIBRARY_DEPENDENCIES
    ${GPUIT_LIBRARY_DEPENDENCIES}
    ${CUDA_LIBRARIES}
    )
ENDIF(CUDA_FOUND)
IF(OPENMP_FOUND)
  SET(GPUIT_LIBRARY_DEPENDENCIES
    ${GPUIT_LIBRARY_DEPENDENCIES}
    ${OPENMP_LIBRARIES}
    )
ENDIF(OPENMP_FOUND)
SET(GPUIT_LIBRARY_DEPENDENCIES
  ${GPUIT_LIBRARY_DEPENDENCIES}
  ${MATH_LIB}
  )
SET(GPUIT_LIBRARIES gpuit ${GPUIT_LIBRARY_DEPENDENCIES})

######################################################
##  FFTW LIBRARY
######################################################
SET(HAVE_FFTW FALSE)
IF(FFTW_FOUND)
SET(HAVE_FFTW TRUE)
ENDIF(FFTW_FOUND)

######################################################
##  BROOK LIBRARY
######################################################
SET(HAVE_BROOK FALSE)
IF(BROOK_FOUND)
SET(HAVE_BROOK TRUE)
ENDIF(BROOK_FOUND)

######################################################
##  CUDA LIBRARY
######################################################
SET(HAVE_CUDA FALSE)
IF(CUDA_FOUND)
SET(HAVE_CUDA TRUE)
ENDIF(CUDA_FOUND)

######################################################
##  ITK LIBRARY
######################################################
SET(HAVE_ITK FALSE)
FIND_PACKAGE(ITK)
IF(ITK_FOUND)
  SET(HAVE_ITK TRUE)
  INCLUDE(${ITK_USE_FILE})
#  SET(ITK_NOREVIEW_VERSION "3.14.0")
  SET(ITK_VERSION "${ITK_VERSION_MAJOR}.${ITK_VERSION_MINOR}.${ITK_VERSION_PATCH}")
#  IF("${ITK_VERSION}" STRLESS "${ITK_NOREVIEW_VERSION}")
#  	SET(ITK_VERSION_314 FALSE)
#  ELSE("${ITK_VERSION}" STRLESS "${ITK_NOREVIEW_VERSION}")
#	SET(ITK_VERSION_314 TRUE)
#  ENDIF("${ITK_VERSION}" STRLESS "${ITK_NOREVIEW_VERSION}")	
ENDIF(ITK_FOUND)

######################################################
##  PANTHEIOS LIBRARY
######################################################
SET(HAVE_PANTHEIOS FALSE)
IF(PANTHEIOS_FOUND)
SET(HAVE_PANTHEIOS TRUE)
ENDIF(PANTHEIOS_FOUND)

######################################################
##  WXWINDOWS LIBRARY
######################################################
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

######################################################
##  DCMTK LIBRARY
######################################################
SET(DCMTK_HAVE_CONFIG_H FALSE)
IF(UNIX)
   SET(DCMTK_HAVE_CONFIG_H TRUE)
ENDIF(UNIX)

######################################################
##  SOURCE FILES (LIBRARIES)
######################################################
SET(GPUIT_LIBRARY_SRC
  bspline_opts.c bspline_opts.h
  bstrlib.c bstrlib.h
  demons.c demons.h
  demons_cpu.c
  demons_misc.c demons_misc.h
  logfile.c logfile.h
  mathutil.h
  readmha.c readmha.h
  volume.c volume.h
  )
IF(PLM_BSPLINE_CXX)
SET(GPUIT_LIBRARY_SRC
	${GPUIT_LIBRARY_SRC}
	bspline.cxx bspline.h)
ELSE(PLM_BSPLINE_CXX)
SET(GPUIT_LIBRARY_SRC
	${GPUIT_LIBRARY_SRC}
	bspline.c bspline.h)
ENDIF(PLM_BSPLINE_CXX)
IF(BROOK_FOUND)
SET(GPUIT_LIBRARY_SRC
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
SET(GPUIT_LIBRARY_SRC
	${GPUIT_LIBRARY_SRC}
	bspline_cuda.cpp
	bspline_cuda_wrap.cpp
	bspline_cuda.cu
	bspline_cuda_kernels.cu
	bspline_cuda.h
	)
ENDIF(CUDA_FOUND)
IF(F2C_LIBRARY)
SET(GPUIT_LIBRARY_SRC
	${GPUIT_LIBRARY_SRC}
	bspline_optimize_lbfgsb.c bspline_optimize_lbfgsb.h
	lbfgsb_2_1.c
	plm_f2c.h plm_f2c_win32.h
#	main__.c
	)
ENDIF(F2C_LIBRARY)
IF(HAVE_BROOK AND BUILD_BSPLINE_BROOK)
SET(GPUIT_LIBRARY_SRC
	${GPUIT_LIBRARY_SRC}
	bspline_brook_kernel_wrap.cpp
    	bspline_brook_kernel.br
#	bspline_brook_kernel.cpp
	toy_kernels.br
	toy_kernels.cpp
	bspline_brook.cpp 
	bspline_brook.h
	)
ENDIF(HAVE_BROOK AND BUILD_BSPLINE_BROOK)

IF(F2C_LIBRARY)
SET(F2C_HELPER_LIBRARY_SRC
	main__.c
	)
ENDIF(F2C_LIBRARY)

SET(PLASTIMATCH_LIBRARY_SRC
	gcs_metric.h gcs_metric.hxx 
	gpuit_bspline.cxx gpuit_bspline.h 
	gpuit_demons.cxx gpuit_demons.h 
	itk_demons.cxx itk_demons.h 
	itk_dicom.cxx itk_dicom.h 
	itk_image.cxx itk_image.h 
	itk_image_cast.cxx itk_image_cast.h 
	itk_pointset.cxx itk_pointset.h
	itk_optim.cxx itk_optim.h 
	itk_registration.cxx itk_registration.h 
	itk_warp.cxx itk_warp.h 
	plm_image.cxx plm_image.h 
	plm_int.h
	plm_parms.cxx 
	plm_stages.cxx 
	plm_registration.h 
	plm_version.h 
	print_and_exit.cxx print_and_exit.h 
	readcxt.cxx readcxt.h 
	resample_mha.cxx resample_mha.h 
	slice_extract.cxx slice_extract.h
	synthetic_mha.cxx synthetic_mha.h
	xform.cxx xform.h 
#	compute_mean_image_main.cxx
	)

######################################################
##  SOURCE FILES (EXECUTABLES)
######################################################
SET(ADJUST_MHA_SRC
	adjust_mha.cxx adjust_mha.h
	)
SET(BSPLINE_SRC
	bspline_main.c 
	)
SET(CMS_TO_CXT_SRC
	cms_to_cxt.cxx
	)
SET(CMS_DOSE_TO_MHA_SRC
	cms_dose_to_mha.c
	)
SET(COMPUTE_DISTANCE_SRC
	compute_distance.cxx compute_distance.h
	closest_point.cxx 
	read_obj.cxx
	)
SET(CUDA_PROBE_SRC
	cuda_probe_main.c
	cuda_probe.cu
	cuda_probe.h
	)
SET(CXT_TO_MHA_SRC
	cxt_to_mha.cxx
	render_polyline.cxx
	render_polyline.h
	)
SET(DEMONS_SRC
	demons_main.c
	demons_opts.c demons_opts.h
	)
SET(DEMONS_CUDA_SRC
	readmha.c readmha.h
	volume.c volume.h
	)
SET(DICE_STATS_SRC
	contour_statistics.cxx contour_statistics.h
	dice_statistics.cxx 
	)
SET(DICOM_RTSS_TO_CXT_SRC 
  dicom_rtss_to_cxt.cxx)
SET(DICOM_UID_SRC
	dicom_uid_main.cxx
	dicom_uid.cxx
	dicom_uid.h
	)
SET(DRR_C_SRC
	drr_c.c
	drr_opts.c drr_opts.h
	)
SET(EXTRACT_CONTOUR_SRC
	extract_contour.cxx
	)
SET(FDK_C_SRC
	fdk_c.c fdk.h
	fdk_opts.c fdk_opts.h
	fdk_utils.c fdk_utils.h
	)
SET(FDK_CUDA_SRC
	fdk_opts.c fdk_opts.h
	fdk_utils.c fdk_utils.h
#	volume.c volume.h
#	readmha.c readmha.h
	fdk_cuda.h fdk.h
	mathutil.h
	fdk_cuda_main.c
	fdk_cuda.cu
	)
SET(FDK_BROOK_SRC
	fdk_brook.cpp fdk.h
	fdk_opts.c fdk_opts.h
	fdk_utils.c fdk_utils.h
	mathutil.h
#	readmha.c readmha.h
#	volume.c volume.h
	)
SET(HND_TO_PFM_SRC
	hnd_to_pfm.c
	)
SET(MASK_MHA_SRC
	mask_mha.cxx
	)
SET(MHA_TO_RTOG_DOSE_SRC
	mha_to_rtog_dose.cxx
	)
SET(MHA_TO_CMS_DOSE_SRC
	mha_to_cms_dose.c
	)
SET(MONDOSHOT_SRC
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
SET(MONDOSHOT_RELAY_SRC
	mondoshot_relay_main.cpp
	)
SET(PATIENT_MASK_SRC
	patient_mask.cxx
	)
SET(POINT_PATH_SRC
	point_path.cxx
	)
SET(PLASTIMATCH_SRC
	plastimatch_main.cxx 
	)
SET(RESAMPLE_MHA_SRC
	resample_mha_main.cxx resample_mha_main.h
	)
SET(COMPUTE_MEAN_IMAGE_SRC
	compute_mean_image_main.cxx
	)
SET(RTOG_TO_MHA_SRC
	rtog_to_mha.cxx
	render_polyline.cxx render_polyline.h
	exchkeys.cxx exchkeys.h
	)
SET(SHUFFLE_MHA_SRC
	shuffle_mha_main.cxx
	)
SET(SYNTHETIC_MHA_SRC
	synthetic_mha_main.cxx synthetic_mha_main.h
	)
SET(TPS_INTERP_SRC
	tps_interp.cxx tps_interp.h
	tps_interp_main.cxx 
	)
SET(VF_COMPARE_SRC
	vf_compare.c
	)
SET(VF_COMPOSE_SRC
	vf_compose.cxx
	)
SET(VF_INVERT_SRC
	vf_invert.cxx
	)
SET(VF_TO_VVF_SRC
	vf_to_vvf.cxx
	)
SET(VF_STATS_SRC
	vf_stats.c
	)
SET(XF_TO_XF_SRC
	xf_to_xf_main.cxx xf_to_xf_main.h
	)
SET(WARP_DIJ_SRC
	warp_dij.cxx
	)
SET(WARP_MHA_SRC
	warp_mha_main.cxx warp_mha_main.h
	)
SET(WARP_POINTSET_SRC
	warp_pointset_main.cxx warp_pointset_main.h
	)

CHECK_FUNCTION_EXISTS(getopt_long HAVE_GETOPT_LONG)
IF(NOT HAVE_GETOPT_LONG)
    SET(ADJUST_MHA_SRC ${ADJUST_MHA_SRC} getopt_long.c getopt.h)
    SET(CXT_TO_MHA_SRC ${CXT_TO_MHA_SRC} getopt_long.c getopt.h)
    SET(DICOM_RTSS_TO_CXT_SRC ${DICOM_RTSS_TO_CXT_SRC} getopt_long.c getopt.h)
    SET(MHA_TO_RTOG_DOSE_SRC ${MHA_TO_RTOG_DOSE_SRC} getopt_long.c getopt.h)
    SET(PATIENT_MASK_SRC ${PATIENT_MASK_SRC} getopt_long.c getopt.h)
    SET(PLASTIMATCH_SRC ${PLASTIMATCH_SRC} getopt_long.c getopt.h)
    SET(RTOG_TO_MHA_SRC ${RTOG_TO_MHA_SRC} getopt_long.c getopt.h)
    SET(RESAMPLE_MHA_SRC ${RESAMPLE_MHA_SRC} getopt_long.c getopt.h)
    SET(COMPUTE_MEAN_IMAGE_SRC ${COMPUTE_MEAN_IMAGE_SRC} getopt_long.c getopt.h)
    SET(SYNTHETIC_MHA_SRC ${SYNTHETIC_MHA_SRC} getopt_long.c getopt.h)
    SET(VF_INVERT_SRC ${VF_INVERT_SRC} getopt_long.c getopt.h)
    SET(WARP_DIJ_SRC ${WARP_DIJ_SRC} getopt_long.c getopt.h)
    SET(WARP_MHA_SRC ${WARP_MHA_SRC} getopt_long.c getopt.h)
    SET(WARP_POINTSET_SRC ${WARP_POINTSET_SRC} getopt_long.c getopt.h)
    SET(XF_TO_XF_SRC ${XF_TO_XF_SRC} getopt_long.c getopt.h)
ENDIF(NOT HAVE_GETOPT_LONG)

######################################################
##  SOURCE FILE PROPERTIES
######################################################
IF(OPENMP_FOUND)
  SET_SOURCE_FILES_PROPERTIES(bspline.c
    PROPERTIES COMPILE_FLAGS ${OPENMP_FLAGS})
ENDIF(OPENMP_FOUND)

######################################################
##  CONFIGURE INCLUDE FILE
######################################################
CONFIGURE_FILE(${CMAKE_CURRENT_SOURCE_DIR}/plm_config.h.cmake 
  ${CMAKE_CURRENT_BINARY_DIR}/plm_config.h)

IF(F2C_LIBRARY)
  CONFIGURE_FILE(${CMAKE_CURRENT_SOURCE_DIR}/plm_f2c.h.cmake 
    ${CMAKE_CURRENT_BINARY_DIR}/plm_f2c.h)
ENDIF(F2C_LIBRARY)

######################################################
##  INCLUDE DIRECTORIES
######################################################
INCLUDE_DIRECTORIES(BEFORE ${CMAKE_CURRENT_SOURCE_DIR})
INCLUDE_DIRECTORIES(BEFORE ${CMAKE_CURRENT_BINARY_DIR})

# IF(F2C_INCLUDE_DIR)
#   INCLUDE_DIRECTORIES(AFTER ${F2C_INCLUDE_DIR})
# ENDIF(F2C_INCLUDE_DIR)

IF(BROOK_FOUND)
  INCLUDE_DIRECTORIES(AFTER ${BROOK_INCLUDE_DIR})
ENDIF(BROOK_FOUND)

IF(CUDA_FOUND)
  # Not sure.  GPUIT had both includes.
  # INCLUDE_DIRECTORIES(AFTER ${CUDA_INCLUDE_DIR})
  INCLUDE_DIRECTORIES(AFTER ${CUDA_INCLUDE_DIR} ${CUDA_SDK_INCLUDE_DIR})
ENDIF(CUDA_FOUND)

IF(DCMTK_FOUND)
  INCLUDE_DIRECTORIES(AFTER ${DCMTK_INCLUDE_DIR})
ENDIF(DCMTK_FOUND)

IF(PANTHEIOS_FOUND)
  INCLUDE_DIRECTORIES(AFTER ${STLSOFT_INCLUDE_DIR})
  INCLUDE_DIRECTORIES(AFTER ${PANTHEIOS_INCLUDE_DIR})
ENDIF(PANTHEIOS_FOUND)

IF(wxWidgets_FOUND)
  INCLUDE_DIRECTORIES(BEFORE ${wxWidgets_INCLUDE_DIRS})
ENDIF(wxWidgets_FOUND)


######################################################
##  FATM LIBRARY
######################################################
ADD_SUBDIRECTORY(fatm/src)


######################################################
##  MAKEFILE TARGETS
######################################################

## http://social.msdn.microsoft.com/Forums/en-US/Vsexpressvc/thread/09ae5e38-8d81-4f3e-bfe8-e6250c31fca6
## SET(ITK_LIBRARIES ${ITK_LIBRARIES} msvcprt)

SET(PLASTIMATCH_LIBS plastimatch1 ${ITK_LIBRARIES} ${GPUIT_LIBRARIES})


# Always built
#ADD_LIBRARY(gpuit STATIC ${GPUIT_LIBRARY_SRC})
ADD_LIBRARY(gpuit ${GPUIT_LIBRARY_SRC})

ADD_EXECUTABLE(analyze_to_mha analyze2mha.c)
ADD_EXECUTABLE(bspline ${BSPLINE_SRC})
ADD_EXECUTABLE(cms_dose_to_mha ${CMS_DOSE_TO_MHA_SRC})
ADD_EXECUTABLE(demons ${DEMONS_SRC})
ADD_EXECUTABLE(drr_c ${DRR_C_SRC})
ADD_EXECUTABLE(fdk_c ${FDK_C_SRC})
ADD_EXECUTABLE(hnd_to_pfm ${HND_TO_PFM_SRC})
ADD_EXECUTABLE(mha_to_raw mha_to_raw.c)
ADD_EXECUTABLE(mha_to_cms_dose mha_to_cms_dose.c)
ADD_EXECUTABLE(raw_to_mha raw_to_mha.c)
ADD_EXECUTABLE(tps_update tps_update.c)
ADD_EXECUTABLE(vf_compare ${VF_COMPARE_SRC})
ADD_EXECUTABLE(vf_to_vvf ${VF_TO_VVF_SRC})
ADD_EXECUTABLE(vf_stats ${VF_STATS_SRC})
ADD_EXECUTABLE(vf3d_to_mha vf3d_to_mha.c)
ADD_EXECUTABLE(vox_to_mha vox_to_mha.c)

# Require brook
IF(BROOK_FOUND)
  BRCC_FILE(fdk_brook_kernel.br)
  SET_SOURCE_FILES_PROPERTIES(fdk_brook.cpp PROPERTIES
    OBJECT_DEPENDS "fdk_brook_kernel.cpp")
  ADD_EXECUTABLE(fdk_brook ${FDK_BROOK_SRC} fdk_brook_kernel.br)

  BRCC_FILE(convolution_brook_kernel.br)
  BRCC_FILE(demons_brook_kernel.br)
  SET_SOURCE_FILES_PROPERTIES(demons_brook_kernel_wrap.cpp PROPERTIES
    OBJECT_DEPENDS "demons_brook_kernel.cpp")
  SET_SOURCE_FILES_PROPERTIES(convolution_brook_kernel_wrap.cpp PROPERTIES
    OBJECT_DEPENDS "convolution_brook_kernel.cpp")

  BRCC_FILE(bspline_brook_kernel.br)
  BRCC_FILE(toy_kernels.br)
  SET_SOURCE_FILES_PROPERTIES(bspline_brook.cpp PROPERTIES
    OBJECT_DEPENDS "bspline_brook_kernel.cpp")
  SET_SOURCE_FILES_PROPERTIES(toy_kernels.cpp PROPERTIES
    OBJECT_DEPENDS "toy_kernels.cpp")
ENDIF(BROOK_FOUND)

# Require cuda
IF(BUILD_DEMONS_CUDA)
IF(CUDA_FOUND)
  NVCC_FILE(demons_cuda.cu)
  SET_SOURCE_FILES_PROPERTIES(demons_cuda.cu PROPERTIES
    OBJECT_DEPENDS "demons_cuda_kernel.cu")
  ADD_EXECUTABLE(demons_cuda ${DEMONS_CUDA_SRC} demons_cuda.obj)
ENDIF(CUDA_FOUND)
ENDIF(BUILD_DEMONS_CUDA)

IF(CUDA_FOUND)
  NVCC_FILE(fdk_cuda.cu)
  SET_SOURCE_FILES_PROPERTIES(fdk_cuda_wrap.cpp
    PROPERTIES OBJECT_DEPENDS "fdk_cuda.cu.cpp")
  #ADD_EXECUTABLE(fdk_cuda ${FDK_CUDA_SRC} ${CUDA_LIBRARIES} fdk_cuda_wrap.cpp)
  ADD_EXECUTABLE(fdk_cuda ${FDK_CUDA_SRC} fdk_cuda_wrap.cpp)
ENDIF(CUDA_FOUND)

IF(CUDA_FOUND)
  NVCC_FILE(cuda_probe.cu)
  SET_SOURCE_FILES_PROPERTIES(cuda_probe_wrap.cpp
    PROPERTIES OBJECT_DEPENDS "cuda_probe.cu.cpp")
  ADD_EXECUTABLE(cuda_probe ${CUDA_PROBE_SRC} ${CUDA_LIBRARIES} 
  			    cuda_probe_wrap.cpp)
ENDIF(CUDA_FOUND)

# Require dcmtk
IF(DCMTK_FOUND)
  ADD_EXECUTABLE(dicom_uid ${DICOM_UID_SRC})
ENDIF(DCMTK_FOUND)

# Require f2c
IF(F2C_LIBRARY AND UNIX)
#  ADD_LIBRARY(f2c_helper STATIC ${F2C_HELPER_LIBRARY_SRC})
  ADD_LIBRARY(f2c_helper ${F2C_HELPER_LIBRARY_SRC})
ENDIF(F2C_LIBRARY AND UNIX)

# Require itk
IF(ITK_FOUND)
  ## Despite claims to the contrary, you are not allowed to have a library
  ## with the same name as the executable.  What happens is the executable 
  ## tries to create a temporary library with the same name.  [MSVC 2005]
  ## That is why the library is called plastimatch1 instead of plastimatch.
  ADD_LIBRARY(plastimatch1 ${PLASTIMATCH_LIBRARY_SRC})

  ADD_EXECUTABLE(adjust_mha ${ADJUST_MHA_SRC})
  ADD_EXECUTABLE(cms_to_cxt ${CMS_TO_CXT_SRC})
  IF(ITK_USE_REVIEW)	
	ADD_EXECUTABLE(compute_mean_image ${COMPUTE_MEAN_IMAGE_SRC})
  ENDIF(ITK_USE_REVIEW)
  ADD_EXECUTABLE(compute_distance ${COMPUTE_DISTANCE_SRC})
  ADD_EXECUTABLE(cxt_to_mha ${CXT_TO_MHA_SRC})
  ADD_EXECUTABLE(dice_stats ${DICE_STATS_SRC})
  ADD_EXECUTABLE(dicom_to_mha dicom_to_mha.cxx)
  ADD_EXECUTABLE(dicom_rtss_to_cxt ${DICOM_RTSS_TO_CXT_SRC})
  IF(ITK_USE_REVIEW)
    ADD_EXECUTABLE(extract_contour ${EXTRACT_CONTOUR_SRC})
  ENDIF(ITK_USE_REVIEW)
  ADD_EXECUTABLE(mask_mha ${MASK_MHA_SRC})
  ADD_EXECUTABLE(merge_vfs merge_vector_fields.cxx)
  ADD_EXECUTABLE(merge2 merge2.cxx)
  ADD_EXECUTABLE(mha_to_analyze mha2analyze.c)
  ADD_EXECUTABLE(mha_to_rtog_dose ${MHA_TO_RTOG_DOSE_SRC})
  ADD_EXECUTABLE(mha_to_vox mha_to_vox.c)
  ADD_EXECUTABLE(point_path ${POINT_PATH_SRC})
  ADD_EXECUTABLE(patient_mask ${PATIENT_MASK_SRC})
  ADD_EXECUTABLE(plastimatch ${PLASTIMATCH_SRC})
  ADD_EXECUTABLE(resample_mha ${RESAMPLE_MHA_SRC})
  ADD_EXECUTABLE(rtog_to_mha ${RTOG_TO_MHA_SRC})
  ADD_EXECUTABLE(shuffle_mha ${SHUFFLE_MHA_SRC})
  ADD_EXECUTABLE(synthetic_mha ${SYNTHETIC_MHA_SRC})
  ADD_EXECUTABLE(tps_interp ${TPS_INTERP_SRC})
  ADD_EXECUTABLE(union_mask union_mask.cxx)
  ADD_EXECUTABLE(vf_compose ${VF_COMPOSE_SRC})
  ADD_EXECUTABLE(vf_invert ${VF_INVERT_SRC})
  ADD_EXECUTABLE(xf_to_xf ${XF_TO_XF_SRC})
  ADD_EXECUTABLE(warp_dij ${WARP_DIJ_SRC})
  ADD_EXECUTABLE(warp_mha ${WARP_MHA_SRC})
  ADD_EXECUTABLE(warp_pointset ${WARP_POINTSET_SRC})
ENDIF(ITK_FOUND)

IF (WIN32 AND NOT CYGWIN AND wxWidgets_FOUND AND DCMTK_FOUND)
  ADD_EXECUTABLE(mondoshot WIN32 ${MONDOSHOT_SRC})
ENDIF (WIN32 AND NOT CYGWIN AND wxWidgets_FOUND AND DCMTK_FOUND)
IF(DCMTK_FOUND)
  #ADD_EXECUTABLE(mondoshot_relay ${MONDOSHOT_RELAY_SRC})
ENDIF(DCMTK_FOUND)

#ADD_EXECUTABLE(ctest_test ctest_test.c)

######################################################
##  LINKING INSTRUCTIONS
######################################################

# OpenMP flags
IF(NOT ${OPENMP_LDFLAGS} STREQUAL "")
  SET_TARGET_PROPERTIES(gpuit PROPERTIES LINK_FLAGS ${OPENMP_LDFLAGS})
  SET_TARGET_PROPERTIES(bspline PROPERTIES LINK_FLAGS ${OPENMP_LDFLAGS})
  IF(ITK_FOUND)
    SET_TARGET_PROPERTIES(plastimatch PROPERTIES LINK_FLAGS ${OPENMP_LDFLAGS})
    SET_TARGET_PROPERTIES(point_path PROPERTIES LINK_FLAGS ${OPENMP_LDFLAGS})
    SET_TARGET_PROPERTIES(tps_interp PROPERTIES LINK_FLAGS ${OPENMP_LDFLAGS})
    SET_TARGET_PROPERTIES(warp_mha PROPERTIES LINK_FLAGS ${OPENMP_LDFLAGS})
    SET_TARGET_PROPERTIES(warp_pointset PROPERTIES LINK_FLAGS ${OPENMP_LDFLAGS})
    SET_TARGET_PROPERTIES(xf_to_xf  PROPERTIES LINK_FLAGS ${OPENMP_LDFLAGS})
  ENDIF(ITK_FOUND)
ENDIF(NOT ${OPENMP_LDFLAGS} STREQUAL "")

# gpuit
TARGET_LINK_LIBRARIES(gpuit ${GPUIT_LIBRARY_DEPENDENCIES})

# plastimatch
IF(ITK_FOUND)
  TARGET_LINK_LIBRARIES(plastimatch1 ${ITK_LIBRARIES} ${GPUIT_LIBRARIES})
ENDIF(ITK_FOUND)

# always built
TARGET_LINK_LIBRARIES(analyze_to_mha)
TARGET_LINK_LIBRARIES(bspline ${GPUIT_LIBRARIES})
TARGET_LINK_LIBRARIES(demons ${GPUIT_LIBRARIES})
TARGET_LINK_LIBRARIES(drr_c ${GPUIT_LIBRARIES})
TARGET_LINK_LIBRARIES(fdk_c ${GPUIT_LIBRARIES})
TARGET_LINK_LIBRARIES(hnd_to_pfm)
TARGET_LINK_LIBRARIES(mha_to_analyze)
TARGET_LINK_LIBRARIES(raw_to_mha)
TARGET_LINK_LIBRARIES(tps_update ${MATH_LIB})
TARGET_LINK_LIBRARIES(vf_compare ${GPUIT_LIBRARIES})
TARGET_LINK_LIBRARIES(vf_to_vvf ${GPUIT_LIBRARIES})
TARGET_LINK_LIBRARIES(vf_stats ${GPUIT_LIBRARIES})
TARGET_LINK_LIBRARIES(vf3d_to_mha)
TARGET_LINK_LIBRARIES(vox_to_mha)

# require brook
IF(BROOK_FOUND)
  TARGET_LINK_LIBRARIES(fdk_brook ${MATH_LIB} ${GPUIT_LIBRARIES}
    ${BROOK_LIBRARIES} ${OPENGL_LIBRARIES} ${DIRECTX_LIBRARIES})
ENDIF(BROOK_FOUND)

# Require cuda
IF(CUDA_FOUND)
  TARGET_LINK_LIBRARIES(fdk_cuda ${GPUIT_LIBRARIES} ${CUDA_LIBRARIES})
  IF(BUILD_DEMONS_CUDA)
    TARGET_LINK_LIBRARIES(demons_cuda ${MATH_LIB} ${CUDA_LIBRARIES})
  ENDIF(BUILD_DEMONS_CUDA)
ENDIF(CUDA_FOUND)

# Require dcmtk
IF(DCMTK_FOUND)
  TARGET_LINK_LIBRARIES(dicom_uid ${DCMTK_LIBRARIES})
ENDIF(DCMTK_FOUND)

# Require itk
IF(ITK_FOUND)
  TARGET_LINK_LIBRARIES(adjust_mha ${PLASTIMATCH_LIBS})
  TARGET_LINK_LIBRARIES(cms_to_cxt ${PLASTIMATCH_LIBS})
  IF(ITK_USE_REVIEW)
 	TARGET_LINK_LIBRARIES(compute_mean_image ${PLASTIMATCH_LIBS})
  ENDIF(ITK_USE_REVIEW)
  TARGET_LINK_LIBRARIES(compute_distance ${PLASTIMATCH_LIBS})
  TARGET_LINK_LIBRARIES(cxt_to_mha ${PLASTIMATCH_LIBS})
  TARGET_LINK_LIBRARIES(dice_stats ${PLASTIMATCH_LIBS})
  TARGET_LINK_LIBRARIES(dicom_to_mha ${PLASTIMATCH_LIBS})
  TARGET_LINK_LIBRARIES(dicom_rtss_to_cxt ${PLASTIMATCH_LIBS})
  IF(ITK_USE_REVIEW)
    TARGET_LINK_LIBRARIES(extract_contour ${PLASTIMATCH_LIBS})
  ENDIF(ITK_USE_REVIEW)
  TARGET_LINK_LIBRARIES(mask_mha ${PLASTIMATCH_LIBS})
  TARGET_LINK_LIBRARIES(merge_vfs ${PLASTIMATCH_LIBS})
  TARGET_LINK_LIBRARIES(merge2 ${ITK_LIBRARIES})
  TARGET_LINK_LIBRARIES(mha_to_rtog_dose ${PLASTIMATCH_LIBS})
  TARGET_LINK_LIBRARIES(point_path ${PLASTIMATCH_LIBS})
  TARGET_LINK_LIBRARIES(patient_mask ${PLASTIMATCH_LIBS})
  TARGET_LINK_LIBRARIES(plastimatch ${PLASTIMATCH_LIBS} ITKStatistics)
  TARGET_LINK_LIBRARIES(resample_mha ${PLASTIMATCH_LIBS})
  TARGET_LINK_LIBRARIES(rtog_to_mha ${ITK_LIBRARIES})
  TARGET_LINK_LIBRARIES(synthetic_mha ${PLASTIMATCH_LIBS})
  TARGET_LINK_LIBRARIES(shuffle_mha ${PLASTIMATCH_LIBS})
  TARGET_LINK_LIBRARIES(union_mask ${ITK_LIBRARIES})
  TARGET_LINK_LIBRARIES(tps_interp ${PLASTIMATCH_LIBS})
  TARGET_LINK_LIBRARIES(vf_compose ${PLASTIMATCH_LIBS})
  TARGET_LINK_LIBRARIES(vf_invert ${PLASTIMATCH_LIBS})
  TARGET_LINK_LIBRARIES(xf_to_xf ${PLASTIMATCH_LIBS})
  TARGET_LINK_LIBRARIES(warp_dij ${PLASTIMATCH_LIBS})
  TARGET_LINK_LIBRARIES(warp_mha ${PLASTIMATCH_LIBS})
  TARGET_LINK_LIBRARIES(warp_pointset ${PLASTIMATCH_LIBS})
ENDIF(ITK_FOUND)

IF (WIN32 AND NOT CYGWIN AND wxWidgets_FOUND AND DCMTK_FOUND)
  TARGET_LINK_LIBRARIES(mondoshot ${wxWidgets_LIBRARIES} 
    ${DCMTK_LIBRARIES})
ENDIF (WIN32 AND NOT CYGWIN AND wxWidgets_FOUND AND DCMTK_FOUND)
IF(DCMTK_FOUND)
  #TARGET_LINK_LIBRARIES(mondoshot_relay ${DCMTK_LIBRARIES})
ENDIF(DCMTK_FOUND)

######################################################
##  INSTALL
######################################################
# Always built
#INSTALL(TARGETS gpuit DESTINATION lib)
INSTALL(TARGETS bspline DESTINATION bin)
INSTALL(TARGETS cms_dose_to_mha DESTINATION bin)
INSTALL(TARGETS demons DESTINATION bin)
INSTALL(TARGETS drr_c DESTINATION bin)
INSTALL(TARGETS fdk_c DESTINATION bin)
INSTALL(TARGETS hnd_to_pfm DESTINATION bin)
INSTALL(TARGETS mha_to_cms_dose DESTINATION bin)
INSTALL(TARGETS mha_to_raw DESTINATION bin)
INSTALL(FILES volume.h readmha.h demons.h DESTINATION include/gpuit)

# Require brook
IF(BROOK_FOUND)
  INSTALL(TARGETS fdk_brook DESTINATION bin)
ENDIF(BROOK_FOUND)

# Require dcmtk
IF(DCMTK_FOUND)
  INSTALL(TARGETS dicom_uid DESTINATION bin)
ENDIF(DCMTK_FOUND)

# Require itk
IF(ITK_FOUND)
  INSTALL(TARGETS adjust_mha DESTINATION bin)
  INSTALL(TARGETS analyze_to_mha DESTINATION bin)
  INSTALL(TARGETS cms_to_cxt DESTINATION bin)
  IF(ITK_USE_REVIEW)
	INSTALL(TARGETS compute_mean_image DESTINATION bin)  
  ENDIF(ITK_USE_REVIEW)
  INSTALL(TARGETS compute_distance DESTINATION bin)
  INSTALL(TARGETS cxt_to_mha DESTINATION bin)
  INSTALL(TARGETS compute_distance DESTINATION bin)
  INSTALL(TARGETS dice_stats DESTINATION bin)
  INSTALL(TARGETS dicom_to_mha DESTINATION bin)
  INSTALL(TARGETS dicom_rtss_to_cxt DESTINATION bin)
  IF(ITK_USE_REVIEW)
    INSTALL(TARGETS extract_contour DESTINATION bin)
  ENDIF(ITK_USE_REVIEW)
  INSTALL(TARGETS mask_mha DESTINATION bin)
  INSTALL(TARGETS merge_vfs DESTINATION bin)
  INSTALL(TARGETS merge2 DESTINATION bin)
  INSTALL(TARGETS mha_to_analyze DESTINATION bin)
  INSTALL(TARGETS mha_to_rtog_dose DESTINATION bin)
  INSTALL(TARGETS mha_to_vox DESTINATION bin)
  INSTALL(TARGETS point_path DESTINATION bin)
  INSTALL(TARGETS patient_mask DESTINATION bin)
  INSTALL(TARGETS raw_to_mha DESTINATION bin)
  INSTALL(TARGETS plastimatch DESTINATION bin)
  INSTALL(TARGETS plastimatch1 DESTINATION lib)
  INSTALL(TARGETS resample_mha DESTINATION bin)
  INSTALL(TARGETS rtog_to_mha DESTINATION bin)
  INSTALL(TARGETS synthetic_mha DESTINATION bin)
  INSTALL(TARGETS shuffle_mha DESTINATION bin)
  INSTALL(TARGETS tps_interp DESTINATION bin)
  INSTALL(TARGETS tps_update DESTINATION bin)
  INSTALL(TARGETS union_mask DESTINATION bin)
  INSTALL(TARGETS vf_compare DESTINATION bin)
  INSTALL(TARGETS vf_to_vvf DESTINATION bin)
  INSTALL(TARGETS vf_stats DESTINATION bin)
  INSTALL(TARGETS vf3d_to_mha DESTINATION bin)
  INSTALL(TARGETS vox_to_mha DESTINATION bin)
  INSTALL(TARGETS vf_compose DESTINATION bin)
  INSTALL(TARGETS vf_invert DESTINATION bin)
  INSTALL(TARGETS xf_to_xf DESTINATION bin)
  INSTALL(TARGETS warp_dij DESTINATION bin)
  INSTALL(TARGETS warp_mha DESTINATION bin)
  INSTALL(TARGETS warp_pointset DESTINATION bin)

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
    slicer3_install_plugins(plastimatch1)
    #slicer3_install_plugins(gpuit)
  endif (Slicer3_DIR)
ENDIF(ITK_FOUND)

INSTALL(FILES 
  COPYRIGHT.TXT
  INSTALL.TXT
  LICENSE.TXT
  README.TXT
  README.BROOK.TXT
  REGISTRATION.TXT
  DESTINATION doc)

IF (WIN32 AND NOT CYGWIN AND wxWidgets_FOUND AND DCMTK_FOUND)
  INSTALL(TARGETS mondoshot DESTINATION bin)
ENDIF (WIN32 AND NOT CYGWIN AND wxWidgets_FOUND AND DCMTK_FOUND)
IF(DCMTK_FOUND)
  #INSTALL(TARGETS mondoshot_relay DESTINATION bin)
ENDIF(DCMTK_FOUND)

######################################################
##  Test executables
###################################################### 	 
# Test executable -- openmp
IF(BUILD_OPENMP_TEST)
  IF(OPENMP_FOUND)
    ADD_EXECUTABLE(openmp_test openmp_test.c)
    TARGET_LINK_LIBRARIES(openmp_test ${MATH_LIB} ${OPENMP_LIBRARIES})
    SET_TARGET_PROPERTIES(openmp_test PROPERTIES COMPILE_FLAGS ${OPENMP_FLAGS})
    MESSAGE(STATUS "OPENMP_LDFLAGS = ${OPENMP_LDFLAGS}")
    IF(NOT ${OPENMP_LDFLAGS} STREQUAL "")
      SET_TARGET_PROPERTIES(openmp_test PROPERTIES LINK_FLAGS ${OPENMP_LDFLAGS})
    ENDIF(NOT ${OPENMP_LDFLAGS} STREQUAL "")
  ENDIF(OPENMP_FOUND)
ENDIF(BUILD_OPENMP_TEST)

# Test executable -- cuda
IF(BUILD_CUDA_TEST)
  IF(CUDA_FOUND)
    NVCC_FILE(cuda_test.cu)
    SET_SOURCE_FILES_PROPERTIES(cuda_test_wrap.cpp
      PROPERTIES OBJECT_DEPENDS "cuda_test.cu.cpp")
    ADD_EXECUTABLE(cuda_test cuda_test_wrap.cpp)
    TARGET_LINK_LIBRARIES(cuda_test ${CUDA_LIBRARIES})
    #  MESSAGE(STATUS "CUDA_LDFLAGS = ${CUDA_LDFLAGS}")
    #  IF(NOT ${CUDA_LDFLAGS} STREQUAL "")
    #    SET_TARGET_PROPERTIES(cuda_test PROPERTIES LINK_FLAGS ${CUDA_LDFLAGS})
    #  ENDIF(NOT ${CUDA_LDFLAGS} STREQUAL "")
  ENDIF(CUDA_FOUND)
ENDIF(BUILD_CUDA_TEST)

# CUDA B-Spline
IF(CUDA_FOUND)
  NVCC_FILE(bspline_cuda.cu)
  SET_SOURCE_FILES_PROPERTIES(bspline_cuda_wrap.cpp
    PROPERTIES OBJECT_DEPENDS "bspline_cuda.cu.cpp")
#  MESSAGE(STATUS "CUDA_LDFLAGS = ${CUDA_LDFLAGS}")
#  IF(NOT ${CUDA_LDFLAGS} STREQUAL "")
#    SET_TARGET_PROPERTIES(bspline_cuda PROPERTIES LINK_FLAGS ${CUDA_LDFLAGS})
#  ENDIF(NOT ${CUDA_LDFLAGS} STREQUAL "")
ENDIF(CUDA_FOUND)

######################################################
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
###################################################### 	 
SET(PLM_TESTING_SOURCE_DIR "${CMAKE_SOURCE_DIR}/Testing")
SET(PLM_TESTING_DATA_DIR "${CMAKE_SOURCE_DIR}/Testing/Data")
SET(PLM_TESTING_BUILD_DIR "${CMAKE_BINARY_DIR}/Testing")
IF (WIN32 AND NOT CYGWIN AND NOT MINGW)
  SET(PLM_ITK_LIBRARY_PATH_HACK -DITK_LIBRARY_PATH=${ITK_LIBRARY_DIRS}/Release)
  SET(PLM_PLASTIMATCH_PATH_HACK ${CMAKE_CURRENT_BINARY_DIR}/Release)
  SET(PLM_PLASTIMATCH_TESTING_PATH_HACK ${CMAKE_CURRENT_BINARY_DIR}/Testing/Release)
ELSE (WIN32 AND NOT CYGWIN AND NOT MINGW)
  SET(PLM_ITK_LIBRARY_PATH_HACK "")
  SET(PLM_PLASTIMATCH_PATH_HACK ${CMAKE_CURRENT_BINARY_DIR})
  SET(PLM_PLASTIMATCH_TESTING_PATH_HACK ${CMAKE_CURRENT_BINARY_DIR}/Testing)
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

######################################################
##  CPACK stuff
##  Note: "make package_source" to create tarball
###################################################### 	 
SET(CPACK_GENERATOR "ZIP")
SET(CPACK_SOURCE_GENERATOR "TBZ2")
SET(CPACK_PACKAGE_VERSION_MAJOR "1")
SET(CPACK_PACKAGE_VERSION_MAJOR "1")
SET(CPACK_PACKAGE_VERSION_MINOR "4-beta")
SET(CPACK_PACKAGE_VERSION_PATCH ${PLASTIMATCH_SVN_VERSION})
set(CPACK_SOURCE_IGNORE_FILES
  "/Testing/"
  "/.svn"
  "~$"
)
INCLUDE(CPack)
