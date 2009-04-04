PROJECT(plastimatch)

######################################################
##  Include files
###################################################### 	 
INCLUDE(CheckFunctionExists)
INCLUDE(FindOpenGL)
##INCLUDE(FindPantheios.cmake)
INCLUDE(PLM_FindSubversion.cmake)
INCLUDE(PLM_FindDCMTK.cmake)
INCLUDE(GPUIT.cmake)


######################################################
##  Plastimatch configuration variables
###################################################### 	 
SET(PLM_BUILD_TESTING ON CACHE BOOL "Enable regression testing")
SET(BUILD_OPENMP_TEST OFF CACHE BOOL "Build OpenMP test program")
SET(BUILD_CUDA_TEST OFF CACHE BOOL "Build Cuda test program")
#SET(BUILD_EXTRACT_CONTOUR OFF CACHE BOOL "Build extract contour executable")
SET(BUILD_BSPLINE_BROOK OFF CACHE BOOL "Build brook version of B-splines")
SET(BUILD_DEMONS_CUDA OFF CACHE BOOL "Build cuda version of demons")
#SET(BUILD_BSPLINE_CUDA ON CACHE BOOL "Build cuda version of B-splines")
#SET(BUILD_FDK_CUDA ON CACHE BOOL "Build CUDA version of FDK")
#SET(BUILD_PLASTIMATCH_SLICER OFF CACHE BOOL "Build Slicer front-end for Plastimatch") 	 
SET(PLM_BSPLINE_CXX OFF CACHE BOOL "Use C++ version of bspline code")

SET(PLASTIMATCH_SVN_VERSION "EXPERIMENTAL")

######################################################
##  SLICER FRONT-END 	 
###################################################### 	 
SET(Slicer3_DIR "" CACHE PATH "Build Slicer front-end for Plastimatch")

IF(Slicer3_DIR)
  cmake_minimum_required(VERSION 2.6)

  MESSAGE (STATUS "Using " ${Slicer3_DIR})

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

  # Test it
  #enable_testing()
  #include(CTest)
  #set(UPDATE_TYPE svn)

  # ? 	 
  #slicer3_add_plugins_test(${CLP}Test1 ${CLP} --help)

ELSE(Slicer3_DIR)
  MESSAGE (STATUS "NOT Using Slicer3")
ENDIF(Slicer3_DIR)

## Tell CMake to be quiet
IF(${CMAKE_MAJOR_VERSION}.${CMAKE_MINOR_VERSION} GREATER 2.4)
   cmake_policy(SET CMP0003 NEW)
ENDIF(${CMAKE_MAJOR_VERSION}.${CMAKE_MINOR_VERSION} GREATER 2.4)

######################################################
##  By default, brook builts /MT.  We need to match this.
######################################################
IF(NOT GPUIT_LINK_MSVCRT)
  STRING(REGEX REPLACE "/MD" "/MT" 
	     CMAKE_CXX_FLAGS_DEBUG 
	     "${CMAKE_CXX_FLAGS_DEBUG}")
  STRING(REGEX REPLACE "/MD" "/MT"
	     CMAKE_CXX_FLAGS_MINSIZEREL 
	     "${CMAKE_CXX_FLAGS_MINSIZEREL}")
  STRING(REGEX REPLACE "/MD" "/MT"
	     CMAKE_CXX_FLAGS_RELEASE 
	     "${CMAKE_CXX_FLAGS_RELEASE}")
  STRING(REGEX REPLACE "/MD" "/MT"
	     CMAKE_CXX_FLAGS_RELWITHDEBINFO 
	     "${CMAKE_CXX_FLAGS_RELWITHDEBINFO}")
  STRING(REGEX REPLACE "/MD" "/MT" 
	     CMAKE_C_FLAGS_DEBUG 
	     "${CMAKE_C_FLAGS_DEBUG}")
  STRING(REGEX REPLACE "/MD" "/MT"
	     CMAKE_C_FLAGS_MINSIZEREL 
	     "${CMAKE_C_FLAGS_MINSIZEREL}")
  STRING(REGEX REPLACE "/MD" "/MT"
	     CMAKE_C_FLAGS_RELEASE 
	     "${CMAKE_C_FLAGS_RELEASE}")
  STRING(REGEX REPLACE "/MD" "/MT"
	     CMAKE_C_FLAGS_RELWITHDEBINFO 
	     "${CMAKE_C_FLAGS_RELWITHDEBINFO}")
ENDIF(NOT GPUIT_LINK_MSVCRT)

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
  SET(MY_MATH_LIB -lm)
ELSE(UNIX)
  SET(MY_MATH_LIB)
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
    f2c_helper
    )
ENDIF(F2C_LIBRARY)
IF(CUDA_FOUND)
  SET(GPUIT_LIBRARY_DEPENDENCIES
    ${GPUIT_LIBRARY_DEPENDENCIES}
    ${CUDA_LIBRARIES}
    )
ENDIF(CUDA_FOUND)
SET(GPUIT_LIBRARY_DEPENDENCIES
  ${GPUIT_LIBRARY_DEPENDENCIES}
  ${MY_MATH_LIB}
  )
SET(GPUIT_LIBRARIES gpuit ${GPUIT_LIBRARY_DEPENDENCIES})

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
ENDIF(ITK_FOUND)

######################################################
##  PANTHEIOS LIBRARY
######################################################
SET(HAVE_PANTHEIOS FALSE)
IF(PANTHEIOS_FOUND)
SET(HAVE_PANTHEIOS TRUE)
ENDIF(PANTHEIOS_FOUND)

######################################################
##  SOURCE FILES (LIBRARIES)
######################################################
SET(GPUIT_LIBRARY_SRC
	bspline_opts.c bspline_opts.h
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
	lbfgsb_2_1.c f2c.h
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
	plm_parms.cxx 
	plm_stages.cxx 
	plm_registration.h 
	plm_version.h 
	print_and_exit.cxx print_and_exit.h 
	resample_mha.cxx resample_mha.h 
	slice_extract.cxx slice_extract.h
	synthetic_mha.cxx synthetic_mha.h
	xform.cxx xform.h 
	gregBSplineDeformableTransform.h 
	gregBSplineDeformableTransform.txx 
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
SET(CONTOUR_STATISTICS_SRC
	contour_statistics.cxx contour_statistics.h
	dice_statistics.cxx 
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
IF(BUILD_DEMONS_CUDA)
SET(DEMONS_CUDA_SRC
	readmha.c readmha.h
	volume.c volume.h
	)
ENDIF(BUILD_DEMONS_CUDA)
SET(DICOM_UID_SRC
	dicom_uid_main.cxx
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
	volume.c volume.h
	readmha.c readmha.h
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
	readmha.c readmha.h
	volume.c volume.h
	)
SET(MASK_MHA_SRC
	mask_mha.cxx
	)
SET(MHA_TO_RTOG_DOSE_SRC
	mha_to_rtog_dose.cxx
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
    SET(MHA_TO_RTOG_DOSE_SRC ${MHA_TO_RTOG_DOSE_SRC} getopt_long.c getopt.h)
    SET(PATIENT_MASK_SRC ${PATIENT_MASK_SRC} getopt_long.c getopt.h)
    SET(PLASTIMATCH_SRC ${PLASTIMATCH_SRC} getopt_long.c getopt.h)
    SET(RESAMPLE_MHA_SRC ${RESAMPLE_MHA_SRC} getopt_long.c getopt.h)
    SET(RTOG_TO_MHA_SRC ${RTOG_TO_MHA_SRC} getopt_long.c getopt.h)
    SET(SYNTHETIC_MHA_SRC ${SYNTHETIC_MHA_SRC} getopt_long.c getopt.h)
    SET(VF_INVERT_SRC ${VF_INVERT_SRC} getopt_long.c getopt.h)
    SET(WARP_DIJ_SRC ${WARP_DIJ_SRC} getopt_long.c getopt.h)
    SET(WARP_MHA_SRC ${WARP_MHA_SRC} getopt_long.c getopt.h)
    SET(WARP_POINTSET_SRC ${WARP_POINTSET_SRC} getopt_long.c getopt.h)
    SET(XF_TO_XF_SRC ${XF_TO_XF_SRC} getopt_long.c getopt.h)
ENDIF(NOT HAVE_GETOPT_LONG)

######################################################
##  DCMTK CONFIGURATION
######################################################
SET(DCMTK_HAVE_CONFIG_H FALSE)
IF(UNIX)
   SET(DCMTK_HAVE_CONFIG_H TRUE)
ENDIF(UNIX)

######################################################
##  CONFIGURE INCLUDE FILE
######################################################
CONFIGURE_FILE(${CMAKE_CURRENT_SOURCE_DIR}/plm_config.h.cmake ${CMAKE_CURRENT_BINARY_DIR}/plm_config.h)

######################################################
##  INCLUDE DIRECTORIES
######################################################
INCLUDE_DIRECTORIES(BEFORE ${CMAKE_CURRENT_SOURCE_DIR})
INCLUDE_DIRECTORIES(BEFORE ${CMAKE_CURRENT_BINARY_DIR})

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

######################################################
##  MAKEFILE TARGETS
######################################################
SET(PLASTIMATCH_LIBS plastimatch1 ${ITK_LIBRARIES} ${GPUIT_LIBRARIES})

# Always built
#ADD_LIBRARY(gpuit STATIC ${GPUIT_LIBRARY_SRC})
ADD_LIBRARY(gpuit ${GPUIT_LIBRARY_SRC})

ADD_EXECUTABLE(analyze_to_mha analyze2mha.c)
ADD_EXECUTABLE(bspline ${BSPLINE_SRC})
ADD_EXECUTABLE(demons ${DEMONS_SRC})
ADD_EXECUTABLE(drr_c ${DRR_C_SRC})
ADD_EXECUTABLE(fdk_c ${FDK_C_SRC})
ADD_EXECUTABLE(mha_to_raw mha_to_raw.c)
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
  ADD_EXECUTABLE(fdk_cuda ${FDK_CUDA_SRC} ${CUDA_LIBRARIES} fdk_cuda_wrap.cpp)
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
IF(F2C_LIBRARY)
#  ADD_LIBRARY(f2c_helper STATIC ${F2C_HELPER_LIBRARY_SRC})
  ADD_LIBRARY(f2c_helper ${F2C_HELPER_LIBRARY_SRC})
ENDIF(F2C_LIBRARY)

# Require itk
IF(ITK_FOUND)

  ## Despite claims to the contrary, you are not allowed to have a library
  ## with the same name as the executable.  What happens is the executable 
  ## tries to create a temporary library with the same name.  [MSVC 2005]
  ## That is why the library is called plastimatch1 instead of plastimatch.
  ADD_LIBRARY(plastimatch1 ${PLASTIMATCH_LIBRARY_SRC})
  TARGET_LINK_LIBRARIES(plastimatch1 ${ITK_LIBRARIES} ${GPUIT_LIBRARIES})

  ADD_EXECUTABLE(adjust_mha ${ADJUST_MHA_SRC})
  ADD_EXECUTABLE(contour_statistics ${CONTOUR_STATISTICS_SRC})
  ADD_EXECUTABLE(cxt_to_mha ${CXT_TO_MHA_SRC})
  ADD_EXECUTABLE(dicom_to_mha dicom_to_mha.cxx)
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

#ADD_EXECUTABLE(ctest_test ctest_test.c)

######################################################
##  LINKING INSTRUCTIONS
######################################################

# Always built
TARGET_LINK_LIBRARIES(analyze_to_mha)
TARGET_LINK_LIBRARIES(bspline ${GPUIT_LIBRARIES})
TARGET_LINK_LIBRARIES(demons ${GPUIT_LIBRARIES})
TARGET_LINK_LIBRARIES(drr_c ${GPUIT_LIBRARIES})
TARGET_LINK_LIBRARIES(fdk_c ${GPUIT_LIBRARIES})
TARGET_LINK_LIBRARIES(mha_to_analyze)
TARGET_LINK_LIBRARIES(raw_to_mha)
TARGET_LINK_LIBRARIES(tps_update ${MY_MATH_LIB})
TARGET_LINK_LIBRARIES(vf_compare ${GPUIT_LIBRARIES})
TARGET_LINK_LIBRARIES(vf_to_vvf ${GPUIT_LIBRARIES})
TARGET_LINK_LIBRARIES(vf_stats ${GPUIT_LIBRARIES})
TARGET_LINK_LIBRARIES(vf3d_to_mha)
TARGET_LINK_LIBRARIES(vox_to_mha)

# Require brook
IF(BROOK_FOUND)
  TARGET_LINK_LIBRARIES(fdk_brook ${MY_MATH_LIB} ${BROOK_LIBRARIES} ${OPENGL_LIBRARIES} ${DIRECTX_LIBRARIES})
ENDIF(BROOK_FOUND)

# Require cuda
IF(BUILD_DEMONS_CUDA)
  IF(CUDA_FOUND)
    TARGET_LINK_LIBRARIES(demons_cuda ${MY_MATH_LIB} ${CUDA_LIBRARIES})
  ENDIF(CUDA_FOUND)
ENDIF(BUILD_DEMONS_CUDA)

# Require dcmtk
IF(DCMTK_FOUND)
  TARGET_LINK_LIBRARIES(dicom_uid ${DCMTK_LIBRARIES})
ENDIF(DCMTK_FOUND)

# Require itk
IF(ITK_FOUND)
  TARGET_LINK_LIBRARIES(adjust_mha ${PLASTIMATCH_LIBS})
  TARGET_LINK_LIBRARIES(contour_statistics ${PLASTIMATCH_LIBS})
  TARGET_LINK_LIBRARIES(cxt_to_mha ${PLASTIMATCH_LIBS})
  TARGET_LINK_LIBRARIES(dicom_to_mha ${PLASTIMATCH_LIBS})
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

######################################################
##  INSTALL
######################################################
# Always built
#INSTALL(TARGETS gpuit DESTINATION lib)
INSTALL(TARGETS bspline DESTINATION bin)
INSTALL(TARGETS demons DESTINATION bin)
INSTALL(TARGETS drr_c DESTINATION bin)
INSTALL(TARGETS fdk_c DESTINATION bin)
#INSTALL(TARGETS mha_to_raw DESTINATION bin)
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
  INSTALL(TARGETS contour_statistics DESTINATION bin)
  INSTALL(TARGETS cxt_to_mha DESTINATION bin)
  INSTALL(TARGETS dicom_to_mha DESTINATION bin)
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

  if (Slicer3_DIR)
    slicer3_install_plugins(plastimatch1)
    #slicer3_install_plugins(gpuit)
  endif (Slicer3_DIR)
ENDIF(ITK_FOUND)

######################################################
##  Test executables
###################################################### 	 
# Test executable -- openmp
IF(BUILD_OPENMP_TEST)
IF(OPENMP_FOUND)
  ADD_EXECUTABLE(openmp_test openmp_test.c)
  TARGET_LINK_LIBRARIES(openmp_test ${OPENMP_LIBRARIES})
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
IF (WIN32 AND NOT CYGWIN)
  SET(PLM_ITK_LIBRARY_PATH_HACK -DITK_LIBRARY_PATH=${ITK_LIBRARY_DIRS}/Release)
  SET(PLM_PLASTIMATCH_PATH_HACK ${CMAKE_CURRENT_BINARY_DIR}/Release)
  SET(PLM_PLASTIMATCH_TESTING_PATH_HACK ${CMAKE_CURRENT_BINARY_DIR}/Testing/Release)
ELSE (WIN32 AND NOT CYGWIN)
  SET(PLM_ITK_LIBRARY_PATH_HACK "")
  SET(PLM_PLASTIMATCH_PATH_HACK ${CMAKE_CURRENT_BINARY_DIR})
  SET(PLM_PLASTIMATCH_TESTING_PATH_HACK ${CMAKE_CURRENT_BINARY_DIR}/Testing)
ENDIF (WIN32 AND NOT CYGWIN)

MACRO(PLM_ADD_TEST TESTNAME WORKING_DIR PLM_TEST_COMMAND PARMS)
  ADD_TEST (${TESTNAME} ${CMAKE_COMMAND} 
    -DWORKING_DIR=${WORKING_DIR}
    ${PLM_ITK_LIBRARY_PATH_HACK}
    -DPLM_TEST_COMMAND=${PLM_TEST_COMMAND}
    -DPARMS=${PARMS}
    -P ${CMAKE_SOURCE_DIR}/RUN_CTEST.cmake
    )
ENDMACRO(PLM_ADD_TEST)

IF(PLM_BUILD_TESTING)
  ENABLE_TESTING()
  ADD_SUBDIRECTORY(Testing)
  PLM_ADD_TEST (
    "plastimatch_usage_1" 
    ""
    ${PLM_PLASTIMATCH_PATH_HACK}/plastimatch
    ""
    )
  if (0)
  PLM_ADD_TEST (
    "plastimatch_y1" 
    "C:/gcs6/idata/synth_mse"
    ${PLM_PLASTIMATCH_TESTING_PATH_HACK}/plastimatch
    "parms.txt"
    )
  endif (0)
ENDIF(PLM_BUILD_TESTING)

######################################################
##  CPACK stuff
###################################################### 	 
SET(CPACK_PACKAGE_VERSION_MAJOR "1")
SET(CPACK_PACKAGE_VERSION_MINOR "4")
SET(CPACK_PACKAGE_VERSION_PATCH "0")
INCLUDE(CPack)
