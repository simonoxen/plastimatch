##-----------------------------------------------------------------------------
##  See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
##-----------------------------------------------------------------------------
project (src_plastimatch_segment)

configure_file (
    ${CMAKE_CURRENT_SOURCE_DIR}/plmsegment_config.h.in
    ${PLM_BUILD_ROOT}/plmsegment_config.h
)

include_directories (BEFORE ${CMAKE_CURRENT_SOURCE_DIR})

##-----------------------------------------------------------------------------
##  SOURCE FILES
##-----------------------------------------------------------------------------
set (PLMSEGMENT_LIBRARY_DEPENDENCIES 
  plmregister
  plmutil
  plmbase
  plmsys
  ${ITK_LIBRARIES}
  )

set (PLMSEGMENT_LIBRARY_SRC
  autolabel.cxx
  autolabel_task.cxx
  autolabel_thumbnailer.cxx
  autolabel_trainer.cxx
  autolabel_ransac_est.cxx
  dlib_trainer.cxx
  mabs.cxx
  mabs_parms.cxx
  mabs_subject.cxx
  mabs_vote.cxx
  segment_body.cxx
  )

##-----------------------------------------------------------------------------
##  BUILD TARGETS
##-----------------------------------------------------------------------------
plm_add_library (
  plmsegment
  "${PLMSEGMENT_LIBRARY_SRC}" 
  "${PLMSEGMENT_LIBRARY_DEPENDENCIES}"
  ""
  "")

##-----------------------------------------------------------------------------
##  SPECIAL RULES FOR SLICER
##-----------------------------------------------------------------------------
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
