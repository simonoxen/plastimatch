PROJECT(plastimatch)

INCLUDE(CheckFunctionExists)
INCLUDE(FindOpenGL)

INCLUDE(GPUIT.cmake)

SET(BUILD_EXTRACT_CONTOUR OFF CACHE BOOL "Build extract contour executable")

SET(PLASTIMATCH_SLICER OFF CACHE BOOL "Build Slicer front-end for Plastimatch")
IF(PLASTIMATCH_SLICER)
  cmake_minimum_required(VERSION 2.6)
  cmake_policy(SET CMP0003 NEW)
ENDIF(PLASTIMATCH_SLICER)


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

######################################################
##  GPUIT LIBRARY
######################################################
FIND_PACKAGE(GPUIT)
IF(NOT GPUIT_FOUND)
  MESSAGE(FATAL_ERROR
          "Cannot build without GPUIT.  Please set GPUIT_DIR.")
ENDIF(NOT GPUIT_FOUND)
LINK_DIRECTORIES(${GPUIT_LIBRARY_DIRS})

SET(GPUIT_LIBRARIES gpuit)
IF(BROOK_FOUND)
SET(GPUIT_LIBRARIES
	${GPUIT_LIBRARIES}
	${BROOK_LIBRARIES}
	${OPENGL_LIBRARIES}
	${DIRECTX_LIBRARIES}
	)
ENDIF(BROOK_FOUND)
IF(F2C_LIBRARY)
SET(GPUIT_LIBRARIES
	${F2C_LIBRARY}
	${GPUIT_LIBRARIES}
	)
ENDIF(F2C_LIBRARY)

######################################################
##  ITK LIBRARY
######################################################
FIND_PACKAGE(ITK)
IF(ITK_FOUND)
  INCLUDE(${ITK_USE_FILE})
ELSE(ITK_FOUND)
  MESSAGE(FATAL_ERROR
          "Cannot build without ITK.  Please set ITK_DIR.")
ENDIF(ITK_FOUND)

######################################################
##  MATH LIBRARY
######################################################
IF(UNIX)
   SET(MY_MATH_LIB -lm)
ELSE(UNIX)
   SET(MY_MATH_LIB)
ENDIF(UNIX)

######################################################
##  SLICER FRONT-END
######################################################
IF(PLASTIMATCH_SLICER)

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
target_link_libraries(${CLP} ${ITK_LIBRARIES})

# Install it
slicer3_install_plugins(${CLP})

# Test it
#enable_testing()
#include(CTest)
#set(UPDATE_TYPE svn)

slicer3_add_plugins_test(${CLP}Test1 ${CLP} --help)
ENDIF(PLASTIMATCH_SLICER)

######################################################
##  SOURCE FILES
######################################################
SET(COMPOSE_VECTOR_FIELDS_SRC
	compose_vector_fields.cxx
	)
SET(EXTRACT_CONTOUR_SRC
	extract_contour.cxx
	slice_extraction.cxx slice_extraction.h
	#itkContourExtractor2DImageFilter.h
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
SET(PLASTIMATCH_LIBRARY_SRC
	debug.c debug.h 
	gpuit_bspline.cxx gpuit_bspline.h 
	gpuit_demons.cxx gpuit_demons.h 
	itk_demons.cxx itk_demons.h 
	itk_image.cxx itk_image.h 
	gcs_metric.h gcs_metric.hxx 
	itk_optim.cxx itk_optim.h 
	itk_registration.cxx itk_registration.h 
	itk_warp.cxx itk_warp.h 
	print_and_exit.cxx print_and_exit.h 
	rad_image.cxx rad_image.h 
	rad_stages.cxx 
	rad_registration.h 
	resample_mha.cxx resample_mha.h 
	version.h 
	xform.cxx xform.h 
	gregBSplineDeformableTransform.h 
	gregBSplineDeformableTransform.txx 
	)
SET(POINT_PATH_SRC
	point_path.cxx
	)
SET(PLASTIMATCH_SRC
	rad_registration_main.cxx 
	)
SET(RESAMPLE_MHA_SRC
	resample_mha_main.cxx resample_mha_main.h
	)
SET(RTOG_TO_MHA_SRC
	rtog_to_mha.cxx
	render_polyline.cxx render_polyline.h
	exchkeys.cxx exchkeys.h
	)
SET(VF_STATS_SRC
	vf_stats.c
	)
SET(XFORM_TO_VF_SRC
	xform_to_vf_main.cxx xform_to_vf_main.h
	)
SET(WARP_DIJ_SRC
	warp_dij.cxx
	)
SET(WARP_MHA_SRC
	warp_mha_main.cxx warp_mha_main.h
	)

CHECK_FUNCTION_EXISTS(getopt_long HAVE_GETOPT_LONG)
IF(NOT HAVE_GETOPT_LONG)
    SET(MHA_TO_RTOG_DOSE_SRC ${MHA_TO_RTOG_DOSE_SRC} getopt_long.c getopt.h)
    SET(PATIENT_MASK_SRC ${PATIENT_MASK_SRC} getopt_long.c getopt.h)
    SET(PLASTIMATCH_SRC ${PLASTIMATCH_SRC} getopt_long.c getopt.h)
    SET(RESAMPLE_MHA_SRC ${RESAMPLE_MHA_SRC} getopt_long.c getopt.h)
    SET(RTOG_TO_MHA_SRC ${RTOG_TO_MHA_SRC} getopt_long.c getopt.h)
    SET(XFORM_TO_VF_SRC ${XFORM_TO_VF_SRC} getopt_long.c getopt.h)
    SET(WARP_DIJ_SRC ${WARP_DIJ_SRC} getopt_long.c getopt.h)
    SET(WARP_MHA_SRC ${WARP_MHA_SRC} getopt_long.c getopt.h)
ENDIF(NOT HAVE_GETOPT_LONG)


######################################################
##  CONFIGURATION
######################################################
CONFIGURE_FILE(${CMAKE_CURRENT_SOURCE_DIR}/config.h.cmake ${CMAKE_CURRENT_BINARY_DIR}/config.h)


######################################################
##  INCLUDE DIRECTORIES
######################################################
INCLUDE_DIRECTORIES(BEFORE ${CMAKE_CURRENT_BINARY_DIR})
IF(GPUIT_FOUND)
  INCLUDE_DIRECTORIES(AFTER ${GPUIT_INCLUDE_DIRS})
  IF(BROOK_FOUND)
    INCLUDE_DIRECTORIES(AFTER ${BROOK_INCLUDE_DIR})
  ENDIF(BROOK_FOUND)
  IF(CUDA_FOUND)
    INCLUDE_DIRECTORIES(AFTER ${CUDA_INCLUDE_DIR})
  ENDIF(CUDA_FOUND)
ENDIF(GPUIT_FOUND)


######################################################
##  TARGETS
######################################################

## Despite claims to the contrary, you are not allowed to have a library
## with the same name as the executable.  What happens is the executable 
## tries to create a temporary library with the same name.  [MSVC 2005]
ADD_LIBRARY(plastimatch1 ${PLASTIMATCH_LIBRARY_SRC})
## SET_TARGET_PROPERTIES(plastimatch-lib PROPERTIES OUTPUT_NAME plastimatch)

ADD_EXECUTABLE(analyze_to_mha analyze2mha.c)
ADD_EXECUTABLE(compose_vector_fields ${COMPOSE_VECTOR_FIELDS_SRC})
#ADD_EXECUTABLE(cms_to_mha cms_to_mha.cxx)
ADD_EXECUTABLE(dicom_to_mha dicom_to_mha.cxx)
IF(BUILD_EXTRACT_CONTOUR)
ADD_EXECUTABLE(extract_contour ${EXTRACT_CONTOUR_SRC})
ENDIF(BUILD_EXTRACT_CONTOUR)
ADD_EXECUTABLE(mask_mha	${MASK_MHA_SRC})
ADD_EXECUTABLE(merge_vfs merge_vector_fields.cxx)
ADD_EXECUTABLE(merge2 merge2.cxx)
ADD_EXECUTABLE(mha_to_analyze mha2analyze.c)
ADD_EXECUTABLE(mha_to_raw mha_to_raw.c)
ADD_EXECUTABLE(mha_to_rtog_dose  ${MHA_TO_RTOG_DOSE_SRC})
ADD_EXECUTABLE(mha_to_vox mha_to_vox.c)
ADD_EXECUTABLE(point_path ${POINT_PATH_SRC})
ADD_EXECUTABLE(patient_mask ${PATIENT_MASK_SRC})
ADD_EXECUTABLE(raw_to_mha raw_to_mha.c)
ADD_EXECUTABLE(plastimatch ${PLASTIMATCH_SRC})
ADD_EXECUTABLE(resample_mha ${RESAMPLE_MHA_SRC})
ADD_EXECUTABLE(rtog_to_mha ${RTOG_TO_MHA_SRC})
ADD_EXECUTABLE(tps_update tps_update.c)
ADD_EXECUTABLE(union_mask union_mask.cxx)
ADD_EXECUTABLE(vf_stats ${VF_STATS_SRC})
ADD_EXECUTABLE(vf3d_to_mha vf3d_to_mha.c)
ADD_EXECUTABLE(vox_to_mha vox_to_mha.c)
ADD_EXECUTABLE(xform_to_vf ${XFORM_TO_VF_SRC})
ADD_EXECUTABLE(warp_mha ${WARP_MHA_SRC})
ADD_EXECUTABLE(warp_dij ${WARP_DIJ_SRC})

######################################################
##  LINK
######################################################
SET(ITK_LIBS ITKCommon ITKIO ITKNumerics)

TARGET_LINK_LIBRARIES(analyze_to_mha)
#TARGET_LINK_LIBRARIES(cms_to_mha ${ITK_LIBS})
TARGET_LINK_LIBRARIES(compose_vector_fields plastimatch1 ${ITK_LIBS})
TARGET_LINK_LIBRARIES(dicom_to_mha plastimatch1 ${ITK_LIBS})
IF(BUILD_EXTRACT_CONTOUR)
TARGET_LINK_LIBRARIES(extract_contour plastimatch1 ${ITK_LIBS})
ENDIF(BUILD_EXTRACT_CONTOUR)
TARGET_LINK_LIBRARIES(mask_mha plastimatch1 ${ITK_LIBS})
TARGET_LINK_LIBRARIES(merge_vfs ${ITK_LIBS})
TARGET_LINK_LIBRARIES(merge2 ${ITK_LIBS})
TARGET_LINK_LIBRARIES(mha_to_analyze)
TARGET_LINK_LIBRARIES(mha_to_rtog_dose ${GPUIT_LIBRARIES})
TARGET_LINK_LIBRARIES(point_path plastimatch1 ${ITK_LIBS} ${GPUIT_LIBRARIES})
TARGET_LINK_LIBRARIES(patient_mask plastimatch1 ${ITK_LIBS})
TARGET_LINK_LIBRARIES(raw_to_mha)
TARGET_LINK_LIBRARIES(plastimatch plastimatch1 ${ITK_LIBS} ITKStatistics ${GPUIT_LIBRARIES})
TARGET_LINK_LIBRARIES(resample_mha plastimatch1 ${ITK_LIBS})
TARGET_LINK_LIBRARIES(rtog_to_mha ${ITK_LIBS})
TARGET_LINK_LIBRARIES(union_mask ${ITK_LIBS})
TARGET_LINK_LIBRARIES(tps_update ${MY_MATH_LIB})
TARGET_LINK_LIBRARIES(vf_stats ${GPUIT_LIBRARIES})
TARGET_LINK_LIBRARIES(vf3d_to_mha)
TARGET_LINK_LIBRARIES(vox_to_mha)
TARGET_LINK_LIBRARIES(xform_to_vf plastimatch1 ${ITK_LIBS} ${GPUIT_LIBRARIES})
TARGET_LINK_LIBRARIES(warp_dij plastimatch1 ${ITK_LIBS})
TARGET_LINK_LIBRARIES(warp_mha plastimatch1 ${ITK_LIBS} ${GPUIT_LIBRARIES})

######################################################
##  INSTALL
######################################################
INSTALL(TARGETS analyze_to_mha DESTINATION bin)
INSTALL(TARGETS compose_vector_fields DESTINATION bin)
#INSTALL(TARGETS cms_to_mha DESTINATION bin)
INSTALL(TARGETS dicom_to_mha DESTINATION bin)
IF(BUILD_EXTRACT_CONTOUR)
INSTALL(TARGETS extract_contour DESTINATION bin)
ENDIF(BUILD_EXTRACT_CONTOUR)
INSTALL(TARGETS mask_mha DESTINATION bin)
INSTALL(TARGETS merge_vfs DESTINATION bin)
INSTALL(TARGETS merge2 DESTINATION bin)
INSTALL(TARGETS mha_to_analyze DESTINATION bin)
INSTALL(TARGETS mha_to_raw DESTINATION bin)
INSTALL(TARGETS mha_to_rtog_dose DESTINATION bin)
INSTALL(TARGETS mha_to_vox DESTINATION bin)
INSTALL(TARGETS point_path DESTINATION bin)
INSTALL(TARGETS patient_mask DESTINATION bin)
INSTALL(TARGETS raw_to_mha DESTINATION bin)
INSTALL(TARGETS plastimatch DESTINATION bin)
INSTALL(TARGETS resample_mha DESTINATION bin)
INSTALL(TARGETS rtog_to_mha DESTINATION bin)
INSTALL(TARGETS tps_update DESTINATION bin)
INSTALL(TARGETS union_mask DESTINATION bin)
INSTALL(TARGETS vf_stats DESTINATION bin)
INSTALL(TARGETS vf3d_to_mha DESTINATION bin)
INSTALL(TARGETS vox_to_mha DESTINATION bin)
INSTALL(TARGETS xform_to_vf DESTINATION bin)
INSTALL(TARGETS warp_dij DESTINATION bin)
INSTALL(TARGETS warp_mha DESTINATION bin)

# This is just a test, don't install it
# ADD_EXECUTABLE(testme testme.cxx)
# TARGET_LINK_LIBRARIES(testme ITKCommon ITKIO ITKNumerics)
