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
  gamma_analysis.cxx
  itk_crop.cxx
  itk_gabor.cxx
  itk_mask.cxx
  itk_warp.cxx
  landmark_diff.cxx
  plm_warp.cxx
  rasterize_slice.cxx
  slice_extract.cxx
  ss_img_extract.cxx
  )

set (PLMUTIL_LIBRARY_DEPENDENCIES 
  plmbase
  plmsys
  )

##-----------------------------------------------------------------------------
##  BUILD TARGETS
##-----------------------------------------------------------------------------
plm_add_library (
  plmutil
  "${PLMUTIL_LIBRARY_SRC}" 
  "${PLMUTIL_LIBRARY_DEPENDENCIES}"
  "")

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
