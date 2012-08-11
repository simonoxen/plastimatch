##-----------------------------------------------------------------------------
##  See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
##-----------------------------------------------------------------------------
project (src_plastimatch_dose)

configure_file (
    ${CMAKE_CURRENT_SOURCE_DIR}/plmdose_config.h.in
    ${PLM_BUILD_ROOT}/plmdose_config.h
)

include_directories (BEFORE ${CMAKE_CURRENT_SOURCE_DIR})

##-----------------------------------------------------------------------------
##  SOURCE FILES
##-----------------------------------------------------------------------------
set (PLMDOSE_LIBRARY_SRC
  aperture.cxx
  bragg_curve.cxx
  proton_beam.cxx
  proton_dose.cxx
  proton_parms.cxx
  proton_scene.cxx
  wed_parms.cxx
  )

if (NOT BUILD_DEBIAN)
  set (PLMDOSE_LIBRARY_SRC
    ${PLMDOSE_LIBRARY_SRC}
    bragg_curve.cxx
    )
endif ()

set (PLMDOSE_LIBRARY_DEPENDENCIES 
  plmbase
  plmsys
  )

if (NOT BUILD_DEBIAN)
  set (PLMDOSE_LIBRARY_DEPENDENCIES 
    ${PLMDOSE_LIBRARY_DEPENDENCIES} 
    specfun)
endif ()

set (PLMDOSE_LIBRARY_DEPENDENCIES 
  ${PLMDOSE_LIBRARY_DEPENDENCIES} 
  ${PLM_FORTRAN_LIBRARIES}
  )

##-----------------------------------------------------------------------------
##  BUILD TARGETS
##-----------------------------------------------------------------------------
plm_add_library (
  plmdose
  "${PLMDOSE_LIBRARY_SRC}" 
  "${PLMDOSE_LIBRARY_DEPENDENCIES}"
  ""
  "")

##-----------------------------------------------------------------------------
##  SPECIAL RULES FOR SLICER
##-----------------------------------------------------------------------------
if (SLICER_FOUND)
  if (WIN32)
    plm_slicer_copy_dll (plastimatch_slicer_copy_plmdose
      "${INDIR}/plmdose.dll"
      "${OUTDIR}/plmdose.dll"
      "${QTOUTDIR}/plmdose.dll"
      plmdose
    )
  endif ()
endif ()
