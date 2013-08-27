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
  bragg_curve.cxx
  ion_beam.cxx
  ion_dose.cxx
  ion_parms.cxx
  ion_plan.cxx
  ion_plan_p.cxx
  ion_pristine_peak.cxx
  ion_sobp.cxx
  lookup_range.cxx
  lookup_stop.cxx
  wed_parms.cxx
  )

set (PLMDOSE_LIBRARY_DEPENDENCIES 
  plmbase
  plmsys
  )

set (PLMDOSE_LIBRARY_DEPENDENCIES 
  ${PLMDOSE_LIBRARY_DEPENDENCIES} 
  specfun)

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
