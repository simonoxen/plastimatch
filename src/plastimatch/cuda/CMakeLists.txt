##-----------------------------------------------------------------------------
##  See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
##-----------------------------------------------------------------------------
project (src_plastimatch_cuda)

include_directories (BEFORE ${CMAKE_CURRENT_SOURCE_DIR})

##-----------------------------------------------------------------------------
##  SOURCE FILES
##-----------------------------------------------------------------------------
set (PLMCUDA_LIBRARY_SRC
  cuda_kernel_util.cu
  cuda_mem.cu
  cuda_util.cu
  )

##-----------------------------------------------------------------------------
##  LIBRARY DEPENDENCIES
##-----------------------------------------------------------------------------
set (PLMCUDA_LIBRARY_DEPENDENCIES 
  plmsys
  ${CUDA_LIBRARIES}
  )

##-----------------------------------------------------------------------------
##  SPECIAL BUILD RULES: CUDA
##-----------------------------------------------------------------------------
if (CUDA_FOUND)
  if (PLM_USE_GPU_PLUGINS)
    cuda_add_library (plmcuda SHARED
      ${PLMCUDA_LIBRARY_SRC}
      )
    set_target_properties (plmcuda PROPERTIES 
      ARCHIVE_OUTPUT_DIRECTORY "${PLM_BUILD_ROOT}"
      LIBRARY_OUTPUT_DIRECTORY "${PLM_BUILD_ROOT}"
      RUNTIME_OUTPUT_DIRECTORY "${PLM_BUILD_ROOT}")
    target_link_libraries (plmcuda
      ${PLMCUDA_LIBRARY_DEPENDENCIES}
      )
    install (TARGETS plmcuda
      RUNTIME DESTINATION bin
      LIBRARY DESTINATION lib
      )
  else ()
    cuda_compile (CUDA_WRAPPERS ${PLMCUDA_LIBRARY_SRC})
    set (PLMCUDA_LIBRARY_SRC
      ${PLMCUDA_LIBRARY_SRC}
      ${CUDA_WRAPPERS}
      )
    plm_add_library (
      plmcuda
      "${PLMCUDA_LIBRARY_SRC}" 
      "${PLMCUDA_LIBRARY_DEPENDENCIES}"
      "")
  endif ()
endif ()
