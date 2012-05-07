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
  itk_optim.cxx
  itk_registration.cxx
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

set (PLMREGISTER_CUDA_SRC
  ../cuda/cuda_kernel_util.cu
  ../cuda/cuda_mem.cu
  ../cuda/cuda_util.cu
  bspline_cuda.cu 
  bspline_cuda_kernels.h
  demons_cuda.cu
  )

set (PLMREGISTER_CUDA_IFACE_SRC
  demons_misc.cxx
  bspline_cuda.cpp
  )

##-----------------------------------------------------------------------------
##  LIBRARY DEPENDENCIES
##-----------------------------------------------------------------------------
set (PLMREGISTER_LIBRARY_DEPENDENCIES 
  ${LIBLBFGS_LIBRARIES}
  nocedal
  plmbase
  plmsys
  plmutil
  ${PLM_FORTRAN}
  )

if (OPENCL_FOUND)
  set (PLMREGISTER_LIBRARY_DEPENDENCIES 
    ${PLMREGISTER_LIBRARY_DEPENDENCIES}
    plmopencl
    )
endif ()

##-----------------------------------------------------------------------------
##  SPECIAL BUILD RULES: CUDA
##-----------------------------------------------------------------------------
if (CUDA_FOUND)
  if (PLM_USE_GPU_PLUGINS)
    # build CUDA as plmregistercuda
    set (PLMREGISTERCUDA_LIBRARY_SRC
      ${PLMREGISTER_CUDA_IFACE_SRC}
      ${PLMREGISTER_CUDA_SRC}
      )
    cuda_add_library (plmregistercuda SHARED
      ${PLMREGISTERCUDA_LIBRARY_SRC}
      )
    set_target_properties (plmregistercuda PROPERTIES 
      ARCHIVE_OUTPUT_DIRECTORY "${PLM_BUILD_ROOT}"
      LIBRARY_OUTPUT_DIRECTORY "${PLM_BUILD_ROOT}"
      RUNTIME_OUTPUT_DIRECTORY "${PLM_BUILD_ROOT}")
  else ()
    # building CUDA into plmregister
    cuda_compile (CUDA_WRAPPERS ${PLMREGISTER_CUDA_SRC})
    set (PLMREGISTER_LIBRARY_SRC
      ${PLMREGISTER_LIBRARY_SRC}
      ${PLMREGISTER_CUDA_IFACE_SRC}
      ${CUDA_WRAPPERS}
      )
    set (PLMREGISTER_LIBRARY_DEPENDENCIES
      ${PLMREGISTER_LIBRARY_DEPENDENCIES}
      ${CUDA_LIBRARIES}
      )
  endif ()
endif ()

# if building cuda support as a plugin...
if (PLM_USE_GPU_PLUGINS)
  # windows needs to delayload it
  if (WIN32 AND NOT CYGWIN AND NOT MINGW)
    set (PLMREGISTER_LIBRARY_DEPENDENCIES
      ${PLMREGISTER_LIBRARY_DEPENDENCIES}
      plmregistercuda)
    set (PLMREGISTER_LIBRARY_LDFLAGS
      "${PLMREGISTER_LIBRARY_LDFLAGS} /DELAYLOAD:plmregistercuda.dll")
  endif ()
  # and unix needs dlopen()
  if (NOT WIN32 AND LIBDL_FOUND)
    set (PLMREGISTER_LIBRARY_DEPENDENCIES
      ${PLMREGISTER_LIBRARY_DEPENDENCIES}
      -ldl)
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
  "${PLMREGISTER_LIBRARY_LDFLAGS}")

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
    plm_slicer_copy_dll (plastimatch_slicer_copy_plmregistercuda
      "${INDIR}/plmregistercuda.dll"
      "${OUTDIR}/plmregistercuda.dll"
      "${QTOUTDIR}/plmregistercuda.dll"
      plmregistercuda
    )
  endif ()
endif ()



##### 
# JAS 2011.01.13
# New OpenCL plugin stuff (plmopencl)
#if (OPENCL_FOUND)
#  if (PLM_USE_GPU_PLUGINS)      # << Use dynamic loading
#    if (WIN32)
#      # Windows
#      set (GPUIT_LIBRARY_DEPENDENCIES
#        ${GPUIT_LIBRARY_DEPENDENCIES}
#        plmopencl
#        # shared library: cuda extensions
#        # we tell windows to "delay load" this
#      )
#    else ()
#      # Linux
#      set (GPUIT_LIBRARY_DEPENDENCIES
#        ${GPUIT_LIBRARY_DEPENDENCIES}
#        # In linux we use delayload.h to load plmopencl
#        # in the code @ runtime.
#      )
#    endif ()
#  else ()    # << use static linking
#    set (GPUIT_LIBRARY_DEPENDENCIES
#      ${GPUIT_LIBRARY_DEPENDENCIES}
#      ${OPENCL_LIBRARIES}
#    )
#  endif ()
#endif ()
#
#if (WIN32 AND NOT CYGWIN AND NOT MINGW)
#  if (PLM_USE_GPU_PLUGINS)
#    set (GPUIT_LIB_LDFLAGS 
#      "${GPUIT_LIB_LDFLAGS} /DELAYLOAD:plmcuda.dll /DELAYLOAD:plmopencl.dll")
#    set (PLMCUDA_EXE_LDFLAGS 
#      "${PLMCUDA_EXE_LDFLAGS} /DELAYLOAD:plmcuda.dll /DELAYLOAD:plmopencl.dll")
#  else ()
#    set (GPUIT_LIB_LDFLAGS 
#      "${GPUIT_LIB_LDFLAGS} /DELAYLOAD:nvcuda.dll /DELAYLOAD:opencl.dll")
#    set (PLMCUDA_EXE_LDFLAGS 
#      "${PLMCUDA_EXE_LDFLAGS} /DELAYLOAD:nvcuda.dll /DELAYLOAD:opencl.dll")
#  endif ()
#endif ()
