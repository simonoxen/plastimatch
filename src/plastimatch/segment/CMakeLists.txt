##-----------------------------------------------------------------------------
##  See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
##-----------------------------------------------------------------------------
project (src_plastimatch_segment)

configure_file (
    ${CMAKE_CURRENT_SOURCE_DIR}/plmsegment_config.h.in
    ${PLM_BUILD_ROOT}/plmsegment_config.h
)

include_directories (BEFORE ${CMAKE_CURRENT_SOURCE_DIR})

set (PLMSEGMENT_LIBRARY_SRC
  segment_body.cxx
  )

plm_add_library (
  plmsegment
  "${PLMSEGMENT_LIBRARY_SRC}" 
  "${PLMSEGMENT_LIBRARY_DEPENDENCIES}"
  "")

if (SLICER_FOUND)
  if (WIN32)
    plm_slicer_copy_dll (plastimatch_slicer_copy_plmsegment
      "${INDIR}/plmsegment.dll"
      "${OUTDIR}/plmsegment.dll"
      "${QTOUTDIR}/plmsegment.dll"
      plmsegment
    )
  endif ()
endif ()
