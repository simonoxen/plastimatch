##-----------------------------------------------------------------------------
##  See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
##-----------------------------------------------------------------------------
project (src_plastimatch_register)

configure_file (
    ${CMAKE_CURRENT_SOURCE_DIR}/plmregister_config.h.in
    ${PLM_BUILD_ROOT}/plmregister_config.h
)

include_directories (BEFORE ${CMAKE_CURRENT_SOURCE_DIR})
include_directories (BEFORE ${CMAKE_CURRENT_BINARY_DIR})
if (THRUST_INCLUDE_DIR)
  include_directories (AFTER ${THRUST_INCLUDE_DIR})
endif ()

##-----------------------------------------------------------------------------
##  SOURCE FILES
##-----------------------------------------------------------------------------
set (PLMREGISTER_LIBRARY_SRC
  bspline.cxx
  bspline_landmarks.cxx
  bspline_mi.cxx
  bspline_mse.cxx
  bspline_optimize.cxx
  bspline_optimize_lbfgsb.cxx
  bspline_optimize_liblbfgs.cxx
  bspline_optimize_nlopt.cxx
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
  itk_optimizer.cxx
  itk_registration.cxx
  itk_registration_observer.cxx
  itk_tps.cxx
  landmark_warp.cxx 
  plm_parms.cxx
  plm_stages.cxx
  rbf_cluster.cxx
  rbf_gauss.cxx
  rbf_wendland.cxx
  registration_data.cxx
  )

if (OPENCL_FOUND)
  set (PLMREGISTER_LIBRARY_SRC
    ${PLMREGISTER_LIBRARY_SRC}
    demons_opencl.cxx
   )
endif ()

set (PLMREGISTERCUDA_SRC
  bspline_cuda.cu 
  bspline_cuda_kernels.h
  demons_cuda.cu
  )

message (STATUS "CUDA_SDK_ROOT_DIR =  ${CUDA_SDK_ROOT_DIR}")
message (STATUS "CUDA_CUT_INCLUDE_DIR = ${CUDA_CUT_INCLUDE_DIR}")

set (PLMREGISTERCUDA_IFACE_SRC
  bspline_cuda.cxx
  demons_misc.cxx
  viscous_cuda.cxx
  )

if (PLM_BUILD_VISCOUS)
  set (PLMREGISTER_LIBRARY_SRC
    ${PLMREGISTER_LIBRARY_SRC}
    viscous_main.cxx
    )
  set (PLMREGISTERCUDA_SRC
    ${PLMREGISTERCUDA_SRC}
    viscous_compute.cu
    viscous_convolution.cu
    viscous_finalize.cu
    viscous_funcHistogram.cu
    viscous_funcImageDomain.cu
    viscous_initialize.cu
    viscous_main.cu
    )
endif ()

##-----------------------------------------------------------------------------
##  LIBRARY DEPENDENCIES
##-----------------------------------------------------------------------------
set (PLMREGISTER_LIBRARY_DEPENDENCIES 
  ${LIBLBFGS_LIBRARIES}
  nocedal
  plmbase
  plmsys
  plmutil
  ${PLM_FORTRAN_LIBRARIES}
  )

if (CUDA_FOUND)
  set (PLMREGISTER_LIBRARY_DEPENDENCIES 
    ${PLMREGISTER_LIBRARY_DEPENDENCIES}
    plmcuda
    )
endif ()

if (OPENCL_FOUND)
  set (PLMREGISTER_LIBRARY_DEPENDENCIES 
    ${PLMREGISTER_LIBRARY_DEPENDENCIES}
    plmopencl
    )
endif ()

if (NLOPT_FOUND)
  set (PLMREGISTER_LIBRARY_DEPENDENCIES 
    ${PLMREGISTER_LIBRARY_DEPENDENCIES}
    ${NLOPT_LIBRARIES}
    )
endif ()

##-----------------------------------------------------------------------------
##  SPECIAL BUILD RULES: CUDA
##-----------------------------------------------------------------------------
if (CUDA_FOUND)
  if (PLM_USE_GPU_PLUGINS)

    # build CUDA as separate plmregistercuda shared library
    set (PLMREGISTERCUDA_LIBRARY_SRC
      ${PLMREGISTERCUDA_IFACE_SRC}
      ${PLMREGISTERCUDA_SRC}
      )
    plm_add_gpu_plugin_library (plmregistercuda 
      "${PLMREGISTERCUDA_LIBRARY_SRC}")
    target_link_libraries (plmregistercuda 
      plmcuda plmbase ${CUDA_cublas_LIBRARY} ${CUDA_LIBRARIES})

    # set dependencies that plmreconstruct library needs 
    if (WIN32 AND NOT CYGWIN AND NOT MINGW)
      # windows needs to delayload it
      set (PLMREGISTER_LIBRARY_DEPENDENCIES
	${PLMREGISTER_LIBRARY_DEPENDENCIES}
	plmregistercuda)
      set (PLMREGISTER_LIBRARY_LDFLAGS
	"${PLMREGISTER_LIBRARY_LDFLAGS} /DELAYLOAD:plmregistercuda.dll /DELAYLOAD:plmcuda.dll")
    endif ()
    if (NOT WIN32 AND LIBDL_FOUND)
      # and unix needs dlopen()
      set (PLMREGISTER_LIBRARY_DEPENDENCIES
	${PLMREGISTER_LIBRARY_DEPENDENCIES}
	-ldl)
    endif ()
  else ()
    # building CUDA into plmregister
    cuda_compile (CUDA_WRAPPERS ${PLMREGISTERCUDA_SRC})
    set (PLMREGISTER_LIBRARY_SRC
      ${PLMREGISTER_LIBRARY_SRC}
      ${PLMREGISTERCUDA_IFACE_SRC}
      ${CUDA_WRAPPERS}
      )
    set (PLMREGISTER_LIBRARY_DEPENDENCIES
      ${PLMREGISTER_LIBRARY_DEPENDENCIES}
      ${CUDA_cublas_LIBRARY}
      ${CUDA_LIBRARIES}
      )
  endif ()
endif ()

##-----------------------------------------------------------------------------
##  SPECIAL BUILD RULES: OpenMP & SSE2
##-----------------------------------------------------------------------------
if (OPENMP_FOUND)
  set (PLMREGISTER_LIBRARY_LDFLAGS "${OPENMP_LDFLAGS}")
  set_source_files_properties (bspline.cxx
    PROPERTIES COMPILE_FLAGS ${OPENMP_FLAGS})
  set_source_files_properties (bspline_mi.cxx
    PROPERTIES COMPILE_FLAGS ${OPENMP_FLAGS})
  set_source_files_properties (bspline_mse.cxx
    PROPERTIES COMPILE_FLAGS ${OPENMP_FLAGS})
  set_source_files_properties (bspline_regularize_analytic.cxx
    PROPERTIES COMPILE_FLAGS ${OPENMP_FLAGS})
endif ()

# bspline registration benefits from SSE2
if (SSE2_FOUND AND NOT BUILD_AGAINST_SLICER3)
  plm_set_sse2_flags (bspline.cxx bspline_mi.cxx bspline_mse.cxx)
endif ()

##-----------------------------------------------------------------------------
##  BUILD TARGETS
##-----------------------------------------------------------------------------
plm_add_library (
  plmregister
  "${PLMREGISTER_LIBRARY_SRC}" 
  "${PLMREGISTER_LIBRARY_DEPENDENCIES}"
  "${PLMREGISTER_LIBRARY_LDFLAGS}"
  "")

# because plmregistercuda is dynamically loaded (not linked)
# CMake needs to be told explicitly that plmregister
# depends on it, so we tell it explicitly here
if (CUDA_FOUND AND PLM_USE_GPU_PLUGINS)
  add_dependencies (plmregister plmregistercuda)
endif ()

##-----------------------------------------------------------------------------
##  SPECIAL RULES FOR SLICER
##-----------------------------------------------------------------------------
if (SLICER_FOUND)
  if (WIN32)
    plm_slicer_copy_dll (plastimatch_slicer_copy_plmregister
      "${INDIR}/plmregister.dll"
      "${OUTDIR}/plmregister.dll"
      "${QTOUTDIR}/plmregister.dll"
      plmregister
      )
    if (CUDA_FOUND)
      plm_slicer_copy_dll (plastimatch_slicer_copy_plmregistercuda
	"${INDIR}/plmregistercuda.dll"
	"${OUTDIR}/plmregistercuda.dll"
	"${QTOUTDIR}/plmregistercuda.dll"
	plmregistercuda
	)
    endif ()
  endif ()
endif ()
