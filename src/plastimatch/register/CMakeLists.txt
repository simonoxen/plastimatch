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
  demons_state.cxx
  gpuit_bspline.cxx
  gpuit_demons.cxx
  itk_demons.cxx
  itk_optim.cxx
  itk_registration.cxx
  plm_parms.cxx
  plm_stages.cxx
  rbf_cluster.cxx
  rbf_gauss.cxx
  rbf_wendland.cxx
  registration_data.cxx
  )

#set (REG_PLUGIN_SRC
#  bspline_cuda.cpp bspline_cuda.cu bspline_cuda.h
#  cuda_kernel_util.cu cuda_kernel_util.h
#  cuda_mem.cu cuda_mem.h
#  cuda_util.cu cuda_util.h
#  demons_cuda.cu demons_cuda.h
#  demons_misc.cxx
#  )

if (CUDA_FOUND)
    if (NOT PLM_USE_GPU_PLUGINS)
        set (PLM_CUDA_FILES
          bspline_cuda.cu 
          cuda_util.cu
          cuda_kernel_util.cu
          cuda_mem.cu
          demons_cuda.cu
        )
        cuda_compile (CUDA_WRAPPERS ${PLM_CUDA_FILES})
        set (PLMREGISTER_LIBRARY_SRC
            ${PLMREGISTER_LIBRARY_SRC}
            ${CUDA_WRAPPERS}
            bspline_cuda.cpp
            )
    endif ()
endif ()


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
