##-----------------------------------------------------------------------------
##  See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
##-----------------------------------------------------------------------------
project (src_plastimatch_register)

configure_file (
    ${CMAKE_CURRENT_SOURCE_DIR}/plmregister_config.h.in
    ${PLM_BUILD_ROOT}/plmregister_config.h
)

include_directories (BEFORE ${CMAKE_CURRENT_SOURCE_DIR})

set (PLMREGISTER_LIBRARY_SRC
  bspline.cxx
  bspline_landmarks.cxx
  bspline_mi.cxx
  bspline_mse.cxx
  bspline_optimize.cxx
  bspline_optimize_lbfgsb.cxx
  bspline_optimize_liblbfgs.cxx
  bspline_optimize_steepest.cxx
  bspline_regularize.cxx
  bspline_regularize_analytic.cxx
  bspline_regularize_numeric.cxx
  demons.cxx
  demons_cpu.cxx
  demons_misc.cxx
  gpuit_bspline.cxx
  gpuit_demons.cxx
  itk_demons.cxx
  itk_optim.cxx
  itk_registration.cxx
  plm_stages.cxx
  rbf_cluster.cxx
  rbf_gauss.cxx
  rbf_wendland.cxx
  registration_data.cxx
  )

plm_add_library (
  plmregister
  "${PLMREGISTER_LIBRARY_SRC}" 
  "${PLMREGISTER_LIBRARY_DEPENDENCIES}"
  "")

if (SLICER_FOUND)
  if (WIN32)
    plm_slicer_copy_dll (plastimatch_slicer_copy_plmregister
      "${INDIR}/plmregister.dll"
      "${OUTDIR}/plmregister.dll"
      "${QTOUTDIR}/plmregister.dll"
      plmregister
    )
  endif ()
endif ()
