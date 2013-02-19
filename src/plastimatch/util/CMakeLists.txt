##-----------------------------------------------------------------------------
##  See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
##-----------------------------------------------------------------------------
project (src_plastimatch_util)

configure_file (
    ${CMAKE_CURRENT_SOURCE_DIR}/plmutil_config.h.in
    ${PLM_BUILD_ROOT}/plmutil_config.h
)

include_directories (BEFORE ${CMAKE_CURRENT_SOURCE_DIR})

##-----------------------------------------------------------------------------
##  SOURCE FILES
##-----------------------------------------------------------------------------
set (PLMUTIL_LIBRARY_SRC
  bspline_correspond.cxx
  bspline_warp.cxx
  contour_distance.cxx contour_distance.h 
  cxt_extract.cxx
  dice_statistics.cxx
  distance_map.cxx
  diff.cxx
  dvh.cxx 
  gamma_analysis.cxx
  gamma_dose_comparison.cxx gamma_dose_comparison.h
  geometry_chooser.cxx geometry_chooser.h 
  hausdorff_distance.cxx hausdorff_distance.h 
  image_boundary.cxx image_boundary.h
  itk_adjust.cxx
  itk_crop.cxx
  itk_gabor.cxx
  itk_mask.cxx
  itk_scale.cxx
  itk_threshold.cxx
  itk_union.cxx
  itk_warp.cxx
  landmark_diff.cxx
  proj_image_filter.cxx
  plm_patient.cxx
  plm_series.cxx
  plm_study.cxx
  plm_warp.cxx
  rasterize_slice.cxx
  rasterizer.cxx 
  rtds.cxx 
  rtds_dcmtk.cxx
  rtds_gdcm.cxx
  rtds_warp.cxx 
  rtss.cxx
  sift.cxx
  simplify_points.cxx 
  slice_extract.cxx
  ss_img_extract.cxx
  ss_img_stats.cxx 
  synthetic_mha.cxx
  synthetic_vf.cxx
  threshbox.cxx
  vf_invert.cxx
  )
if (FFTW_FOUND)
  set (PLMUTIL_LIBRARY_SRC
    ${PLMUTIL_LIBRARY_SRC}
    ramp_filter.cxx
   )
endif ()

set (PLMUTIL_LIBRARY_HEADERS
  "${CMAKE_BINARY_DIR}/plmutil_config.h"
  )
foreach (ARG ${PLMUTIL_LIBRARY_SRC})
  string (REGEX MATCH ".*\\.h$" TMP "${ARG}")
  if (TMP)
    list (APPEND PLMUTIL_LIBRARY_HEADERS "${TMP}")
  endif ()
endforeach ()

##-----------------------------------------------------------------------------
##  LIBRARY DEPENDENCIES
##-----------------------------------------------------------------------------
set (PLMUTIL_LIBRARY_DEPENDENCIES 
  ${FFTW_LIBRARIES}
  devillard
  plmbase
  plmsys
  )

##-----------------------------------------------------------------------------
##  SPECIAL BUILD RULES: OpenMP & SSE2
##-----------------------------------------------------------------------------
if (OPENMP_FOUND)
  set (PLMUTIL_LIBRARY_LDFLAGS "${OPENMP_LDFLAGS}")
  set_source_files_properties (bspline_warp.cxx
    PROPERTIES COMPILE_FLAGS ${OPENMP_FLAGS})
  set_source_files_properties (dice_statistics.cxx
    PROPERTIES COMPILE_FLAGS ${OPENMP_FLAGS})
  set_source_files_properties (vf_invert.cxx
    PROPERTIES COMPILE_FLAGS ${OPENMP_FLAGS})
endif ()

if (SSE2_FOUND AND NOT BUILD_AGAINST_SLICER3)
  plm_set_sse2_flags (bspline_warp.cxx)
endif ()

##-----------------------------------------------------------------------------
##  BUILD TARGETS
##-----------------------------------------------------------------------------
plm_add_library (
  plmutil
  "${PLMUTIL_LIBRARY_SRC}" 
  "${PLMUTIL_LIBRARY_DEPENDENCIES}"
  "${PLMUTIL_LIBRARY_LDFLAGS}"
  "${PLMUTIL_LIBRARY_HEADERS}")

##-----------------------------------------------------------------------------
##  SPECIAL RULES FOR SLICER
##-----------------------------------------------------------------------------
if (SLICER_FOUND)
  if (WIN32)
    plm_slicer_copy_dll (plastimatch_slicer_copy_plmutil
      "${INDIR}/plmutil.dll"
      "${OUTDIR}/plmutil.dll"
      "${QTOUTDIR}/plmutil.dll"
      plmutil
    )
  endif ()
endif ()
