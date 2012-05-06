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
  bspline_warp.cxx
  cxt_extract.cxx
  diff.cxx
  dvh.cxx 
  gamma_analysis.cxx
  itk_adjust.cxx
  itk_crop.cxx
  itk_gabor.cxx
  itk_mask.cxx
  itk_warp.cxx
  landmark_diff.cxx
  plm_warp.cxx
  rasterize_slice.cxx
  rasterizer.cxx 
  rtds.cxx 
  rtds_dcmtk.cxx
  rtds_gdcm.cxx
  rtds_warp.cxx 
  rtss.cxx
  simplify_points.cxx 
  slice_extract.cxx
  ss_img_extract.cxx
  ss_img_stats.cxx 
  synthetic_mha.cxx
  synthetic_vf.cxx
  threshbox.cxx
  )

set (PLMUTIL_LIBRARY_DEPENDENCIES 
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
  "${PLMUTIL_LIBRARY_LDFLAGS}")

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
