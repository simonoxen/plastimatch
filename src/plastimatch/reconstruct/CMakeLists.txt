##-----------------------------------------------------------------------------
##  See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
##-----------------------------------------------------------------------------
project (src_plastimatch_reconstruct)

configure_file (
    ${CMAKE_CURRENT_SOURCE_DIR}/plmreconstruct_config.h.in
    ${PLM_BUILD_ROOT}/plmreconstruct_config.h
)

include_directories (BEFORE ${CMAKE_CURRENT_SOURCE_DIR})

##-----------------------------------------------------------------------------
##  SOURCE FILES
##-----------------------------------------------------------------------------
set (PLMRECONSTRUCT_LIBRARY_SRC
  bowtie_correction.cxx
  drr.cxx drr.h
  drr_trilin.cxx drr_trilin.h
  fdk.cxx fdk.h
  fdk_util.cxx fdk_util.h
  proj_image.cxx proj_image.h
  proj_image_dir.cxx proj_image_dir.h
  proj_matrix.cxx proj_matrix.h 
  )
if (FFTW_FOUND)
  set (PLMRECONSTRUCT_LIBRARY_SRC
    ${PLMRECONSTRUCT_LIBRARY_SRC}
    ramp_filter.cxx
   )
endif ()
if (OPENCL_FOUND)
  set (PLMRECONSTRUCT_LIBRARY_SRC
    ${PLMRECONSTRUCT_LIBRARY_SRC}
    drr_opencl.cxx
    fdk_opencl.cxx
   )
endif ()

set (PLMRECONSTRUCT_CUDA_SRC
  ../cuda/cuda_util.cu
  drr_cuda.cu
  fdk_cuda.cu
  )

set (PLMRECONSTRUCT_CUDA_IFACE_SRC
  drr_cuda.cpp
  fdk_cuda.cpp
  )

##-----------------------------------------------------------------------------
##  LIBRARY DEPENDENCIES
##-----------------------------------------------------------------------------
set (PLMRECONSTRUCT_LIBRARY_DEPENDENCIES 
  plmbase
  plmsys
  )
if (HAVE_OPENCL)
  set (PLMRECONSTRUCT_LIBRARY_DEPENDENCIES 
    ${PLMRECONSTRUCT_LIBRARY_DEPENDENCIES}
    plmopencl
    )
endif ()

##-----------------------------------------------------------------------------
##  SPECIAL BUILD RULES: CUDA
##-----------------------------------------------------------------------------
if (CUDA_FOUND)
  if (PLM_USE_GPU_PLUGINS)
    # build CUDA as plmreconstructcuda
    set (PLMRECONSTRUCTCUDA_LIBRARY_SRC
      ${PLMRECONSTRUCT_CUDA_IFACE_SRC}
      ${PLMRECONSTRUCT_CUDA_SRC}
    )
    cuda_add_library (plmreconstructcuda SHARED
      ${PLMRECONSTRUCTCUDA_LIBRARY_SRC}
      )
    set_target_properties (plmreconstructcuda PROPERTIES 
      ARCHIVE_OUTPUT_DIRECTORY "${PLM_BUILD_ROOT}"
      LIBRARY_OUTPUT_DIRECTORY "${PLM_BUILD_ROOT}"
      RUNTIME_OUTPUT_DIRECTORY "${PLM_BUILD_ROOT}")
  else ()
    # building CUDA into plmreconstruct
    cuda_compile (CUDA_WRAPPERS ${PLMRECONSTRUCT_CUDA_SRC})
    set (PLMRECONSTRUCT_LIBRARY_SRC
      ${PLMRECONSTRUCT_LIBRARY_SRC}
      ${PLMRECONSTRUCT_CUDA_IFACE_SRC}
      ${CUDA_WRAPPERS}
      )
    set (PLMRECONSTRUCT_LIBRARY_DEPENDENCIES
      ${PLMRECONSTRUCT_LIBRARY_DEPENDENCIES}
      ${CUDA_LIBRARIES}
      )
  endif ()
endif ()

# if building cuda support as a plugin...
if (PLM_USE_GPU_PLUGINS)
  # windows needs to delayload it
  if (WIN32 AND NOT CYGWIN AND NOT MINGW)
    set (PLMRECONSTRUCT_LIBRARY_DEPENDENCIES
      ${PLMRECONSTRUCT_LIBRARY_DEPENDENCIES}
      plmreconstructcuda)
    set (PLMRECONSTRUCT_LIBRARY_LDFLAGS
      "${PLMRECONSTRUCT_LIBRARY_LDFLAGS} /DELAYLOAD:plmreconstructcuda.dll")
  endif ()
  # and unix needs dlopen()
  if (NOT WIN32 AND LIBDL_FOUND)
    set (PLMRECONSTRUCT_LIBRARY_DEPENDENCIES
      ${PLMRECONSTRUCT_LIBRARY_DEPENDENCIES}
      -ldl)
  endif ()
endif ()

##-----------------------------------------------------------------------------
##  SPECIAL BUILD RULES: OpenMP
##-----------------------------------------------------------------------------
if (OPENMP_FOUND)
  set (PLMRECONSTRUCT_LIBRARY_LDFLAGS "${OPENMP_LDFLAGS}")
  set_source_files_properties (drr.cxx
    PROPERTIES COMPILE_FLAGS ${OPENMP_FLAGS})
  set_source_files_properties (fdk.cxx
    PROPERTIES COMPILE_FLAGS ${OPENMP_FLAGS})
endif ()

##-----------------------------------------------------------------------------
##  BUILD TARGETS
##-----------------------------------------------------------------------------
plm_add_library (
  plmreconstruct
  "${PLMRECONSTRUCT_LIBRARY_SRC}" 
  "${PLMRECONSTRUCT_LIBRARY_DEPENDENCIES}"
  "${PLMRECONSTRUCT_LIBRARY_LDFLAGS}")

# because plmreconstructcuda is dynamically loaded (not linked)
# CMake needs to be told explicitly that plmreconstruct
# depends on it, so we tell it explicitly here
if (CUDA_FOUND AND PLM_USE_GPU_PLUGINS)
    add_dependencies (plmreconstruct plmreconstructcuda)
endif ()

##-----------------------------------------------------------------------------
##  SPECIAL RULES FOR SLICER
##-----------------------------------------------------------------------------
if (SLICER_FOUND)
  if (WIN32)
    plm_slicer_copy_dll (plastimatch_slicer_copy_plmreconstruct
      "${INDIR}/plmreconstruct.dll"
      "${OUTDIR}/plmreconstruct.dll"
      "${QTOUTDIR}/plmreconstruct.dll"
      plmreconstruct
    )
  endif ()
endif ()
