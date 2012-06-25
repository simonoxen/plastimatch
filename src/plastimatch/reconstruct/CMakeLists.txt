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
  drr.cxx
  drr_trilin.cxx
  fdk.cxx
  fdk_util.cxx
  )
if (OPENCL_FOUND)
  set (PLMRECONSTRUCT_LIBRARY_SRC
    ${PLMRECONSTRUCT_LIBRARY_SRC}
    drr_opencl.cxx
    fdk_opencl.cxx
   )
endif ()

set (PLMRECONSTRUCT_CUDA_SRC
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
  plmutil
  plmbase
  plmsys
  )
if (CUDA_FOUND)
  set (PLMRECONSTRUCT_LIBRARY_DEPENDENCIES 
    ${PLMRECONSTRUCT_LIBRARY_DEPENDENCIES}
    plmcuda
    )
endif ()
if (OPENCL_FOUND)
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

    # build CUDA as separate plmreconstructcuda shared library
    set (PLMRECONSTRUCTCUDA_LIBRARY_SRC
      ${PLMRECONSTRUCT_CUDA_IFACE_SRC}
      ${PLMRECONSTRUCT_CUDA_SRC}
      )
    plm_add_gpu_plugin_library (plmreconstructcuda 
      "${PLMRECONSTRUCTCUDA_LIBRARY_SRC}")
    target_link_libraries (plmreconstructcuda plmcuda ${CUDA_LIBRARIES})

    # set dependencies that plmreconstruct library needs 
    if (WIN32 AND NOT CYGWIN AND NOT MINGW)
      # windows needs to delayload it
      set (PLMRECONSTRUCT_LIBRARY_DEPENDENCIES
	${PLMRECONSTRUCT_LIBRARY_DEPENDENCIES}
	plmreconstructcuda)
      set (PLMRECONSTRUCT_LIBRARY_LDFLAGS
	"${PLMRECONSTRUCT_LIBRARY_LDFLAGS} /DELAYLOAD:plmreconstructcuda.dll")
    endif ()
    if (NOT WIN32 AND LIBDL_FOUND)
      # and unix needs dlopen()
      set (PLMRECONSTRUCT_LIBRARY_DEPENDENCIES
	${PLMRECONSTRUCT_LIBRARY_DEPENDENCIES}
	-ldl)
    endif ()
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
  "${PLMRECONSTRUCT_LIBRARY_LDFLAGS}"
  "")

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
    if (CUDA_FOUND)
      plm_slicer_copy_dll (plastimatch_slicer_copy_plmreconstructcuda
	"${INDIR}/plmreconstructcuda.dll"
	"${OUTDIR}/plmreconstructcuda.dll"
	"${QTOUTDIR}/plmreconstructcuda.dll"
	plmreconstructcuda
	)
    endif ()
  endif ()
endif ()
