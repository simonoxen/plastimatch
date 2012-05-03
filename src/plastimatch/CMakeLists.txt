##-----------------------------------------------------------------------------
##  Welcome to the Plastimatch CMakeLists.txt file
##-----------------------------------------------------------------------------
##  See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
##-----------------------------------------------------------------------------
project (src_plastimatch)

##-----------------------------------------------------------------------------
##  Local options
##-----------------------------------------------------------------------------

# Plastimatch software configuration options
option (PLM_CONFIG_ALT_DCOS "Use alternative direction cosines rules" OFF)
option (PLM_CONFIG_KEYHOLIZE "Enable RT structure keyholization" OFF)
option (PLM_CONFIG_PREFER_DCMTK 
  "Use DCMTK instead of GDCM for DICOM operations" OFF)
option (PLM_USE_SS_IMAGE_VEC
  "Save structure sets as ITK UCHAR VEC images (in development)" OFF)

##-----------------------------------------------------------------------------
##  Include directories
##-----------------------------------------------------------------------------
include_directories (BEFORE ${CMAKE_BINARY_DIR})
include_directories (BEFORE ${CMAKE_CURRENT_SOURCE_DIR})
# JAS 2012.04.25 !!!!!!!!!!! TEMPORARY !!!!!!!!!!!!!!!!!!!!!!
include_directories (BEFORE ${CMAKE_CURRENT_SOURCE_DIR}/base)
include_directories (BEFORE ${CMAKE_CURRENT_SOURCE_DIR}/cli)
include_directories (BEFORE ${CMAKE_CURRENT_SOURCE_DIR}/dose)
include_directories (BEFORE ${CMAKE_CURRENT_SOURCE_DIR}/qt)
include_directories (BEFORE ${CMAKE_CURRENT_SOURCE_DIR}/cuda)
include_directories (BEFORE ${CMAKE_CURRENT_SOURCE_DIR}/reconstruct)
include_directories (BEFORE ${CMAKE_CURRENT_SOURCE_DIR}/register)
include_directories (BEFORE ${CMAKE_CURRENT_SOURCE_DIR}/script)
include_directories (BEFORE ${CMAKE_CURRENT_SOURCE_DIR}/segment)
include_directories (BEFORE ${CMAKE_CURRENT_SOURCE_DIR}/standalone)
include_directories (BEFORE ${CMAKE_CURRENT_SOURCE_DIR}/sys)
include_directories (BEFORE ${CMAKE_CURRENT_SOURCE_DIR}/util)
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
include_directories (BEFORE ${CMAKE_CURRENT_BINARY_DIR})
include_directories (AFTER ${CMAKE_SOURCE_DIR}/libs/itk-3.20.0)
include_directories (AFTER ${CMAKE_SOURCE_DIR}/libs/nSIFT)
include_directories (AFTER ${BSTRLIB_INCLUDE_DIR})
include_directories (AFTER ${DLIB_INCLUDE_DIR})
include_directories (AFTER ${LUA_INCLUDE_DIR})
include_directories (AFTER ${LIBLBFGS_INCLUDE_DIR})
include_directories (AFTER ${MSINTTYPES_INCLUDE_DIR})
include_directories (AFTER ${RANSAC_INCLUDE_DIRS})

if (CUDA_FOUND)
  include_directories (AFTER ${CUDA_INCLUDE_DIRS})
endif ()
if (DCMTK_FOUND)
  include_directories (AFTER ${DCMTK_INCLUDE_DIR})
endif ()
if (FFTW_FOUND)
  include_directories (BEFORE ${FFTW_INCLUDE_DIR})
  link_directories (${FFTW_DIR})
endif ()
if (NOT HAVE_GETOPT_LONG)
  include_directories (AFTER "${CMAKE_SOURCE_DIR}/libs/getopt")
endif ()
if (MATLAB_FOUND)
  include_directories (AFTER ${MATLAB_INCLUDE_DIRS})
endif ()
if (NLOPT_FOUND)
  include_directories (AFTER ${NLOPT_INCLUDE_DIR})
endif ()
if (OPENCL_FOUND)
  include_directories (BEFORE ${OPENCL_INCLUDE_DIR})
endif ()
if (PANTHEIOS_FOUND)
  include_directories (AFTER ${STLSOFT_INCLUDE_DIR})
  include_directories (AFTER ${PANTHEIOS_INCLUDE_DIR})
endif ()
if (QT4_FOUND)
  if (QT_QTGUI_FOUND)
    include_directories (AFTER ${QT_QTGUI_INCLUDE_DIR})
  endif ()
  if (QT_QTSQL_FOUND)
    include_directories (AFTER ${QT_QTSQL_INCLUDE_DIR})
  endif ()
endif ()
if (SQLITE_FOUND)
  include_directories (AFTER ${SQLITE_INCLUDE_DIR})
endif ()
if (wxWidgets_FOUND)
  include_directories (BEFORE ${wxWidgets_INCLUDE_DIRS})
endif ()

##-----------------------------------------------------------------------------
##  PLMSYS/GPUIT/PLASTIMATCH LIBRARY SOURCE FILES
##-----------------------------------------------------------------------------

#set (PLMCUDA_LIBRARY_SRC
#  bspline_cuda.cpp bspline_cuda.cu bspline_cuda.h
#  cuda_kernel_util.cu cuda_kernel_util.h
#  cuda_mem.cu cuda_mem.h
#  cuda_util.cu cuda_util.h
#  demons_cuda.cu demons_cuda.h
#  register/demons_misc.cxx
#  drr_cuda.cpp drr_cuda.cu drr_cuda.h drr_cuda_p.h 
#  fdk_cuda.cpp fdk_cuda.cu fdk_cuda_p.h fdk_cuda.h
#  )
#
#if (CUDA_FOUND)
#  if (PLM_USE_GPU_PLUGINS)      # << Dynamic loading
#    set (PLMCUDA_LIBRARY_DEPENDENCIES
#      plmsys
#    )
#    cuda_add_library (plmcuda SHARED
#      ${PLMCUDA_LIBRARY_SRC}
#      )
#    set_target_properties (plmcuda PROPERTIES 
#      ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
#      LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
#      RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
#    target_link_libraries (plmcuda
#      ${PLMCUDA_LIBRARY_DEPENDENCIES}
#      )
#    install (TARGETS plmcuda
#      RUNTIME DESTINATION bin
#      LIBRARY DESTINATION lib
#      )
#  else ()        # << Static linking
#    set (GPUIT_LIBRARY_SRC
#      ${GPUIT_LIBRARY_SRC}
#      ${PLMCUDA_LIBRARY_SRC}
#      )
#    set (PLM_CUDA_FILES
#      bspline_cuda.cu 
#      cuda_util.cu
#      cuda_kernel_util.cu
#      cuda_mem.cu
#      demons_cuda.cu
#      drr_cuda.cu
#      fdk_cuda.cu
#      )
#    cuda_compile (CUDA_WRAPPERS ${PLM_CUDA_FILES})
#    set (GPUIT_LIBRARY_SRC
#      ${GPUIT_LIBRARY_SRC}
#      ${CUDA_WRAPPERS}
#      ${PLM_CUDA_FILES}
#      )
#  endif ()
#endif ()


#if (OPENCL_FOUND)
#  # Define OpenCL Plugin (plmopencl)
#  set (PLMOPENCL_LIBRARY_SRC
#    autotune_opencl.cxx autotune_opencl.h
#    opencl_probe.cxx opencl_probe.h opencl_probe.cl
#    opencl_util.cxx opencl_util.h
#    )
#  if (BUILD_SHARED_LIBS)      # << Dynamic loading
#    set (PLMOPENCL_LIBRARY_SRC
#      ${PLMOPENCL_LIBRARY_SRC}
#      )
#    set (PLMOPENCL_LIBRARY_DEPENDENCIES
#      plmsys
#      )
#    # OpenCL Plugin is ALWAYS shared!
#    add_library (plmopencl SHARED
#      ${PLMOPENCL_LIBRARY_SRC}
#      )
#    set_target_properties (plmopencl PROPERTIES 
#      ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
#      LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
#      RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
#    # Be sure to link against libOpenCL and other depends
#    target_link_libraries (plmopencl
#      ${OPENCL_LIBRARIES}
#      ${PLMOPENCL_LIBRARY_DEPENDENCIES}
#      )
#    install (TARGETS plmopencl
#        RUNTIME DESTINATION bin
#        LIBRARY DESTINATION lib
#        )
#  else ()        # << Static linking
#    set (GPUIT_LIBRARY_SRC
#      ${GPUIT_LIBRARY_SRC}
#      ${PLMOPENCL_LIBRARY_SRC}
#      )
#  endif ()
#  set (GPUIT_LIBRARY_SRC
#    ${GPUIT_LIBRARY_SRC}
#    demons_opencl.cxx
#    demons_opencl_p.h
#    demons_opencl.cl
#    drr_opencl.cxx
#    drr_opencl_p.h
#    drr_opencl.h
#    drr_opencl.cl
#    fdk_opencl.cxx 
#    fdk_opencl_p.h
#    fdk_opencl.h
#    fdk_opencl.cl
#    )
#  plm_add_opencl_file (GPUIT_LIBRARY_SRC demons_opencl.cl)
#  plm_add_opencl_file (GPUIT_LIBRARY_SRC drr_opencl.cl)
#  plm_add_opencl_file (GPUIT_LIBRARY_SRC fdk_opencl.cl)
#endif ()


if (F2C_LIBRARY)
  set (F2C_HELPER_LIBRARY_SRC
    main__.c
    )
endif ()

set (PLASTIMATCH1_LIBRARY_SRC
#  plm_clp.cxx plm_clp.h
  )

##-----------------------------------------------------------------------------
##  SOURCE FILES (EXECUTABLES)
##-----------------------------------------------------------------------------
set (PLASTIMATCH_SLICER_BSPLINE_SRC
  plastimatch_slicer_bspline.cxx
  )
set (PLASTIMATCH_SLICER_DICOMRT_IMPORT_SRC
  plastimatch_slicer_dicomrt_import.cxx
  )
set (PLASTIMATCH_SLICER_DICOMRT_EXPORT_SRC
  plastimatch_slicer_dicomrt_export.cxx
  )
set (PLASTIMATCH_SLICER_LANDWARP_SRC
  plastimatch-slicer-landwarp.cxx
  )
set (PLASTIMATCH_SLICER_SYNTH_SRC
  plastimatch_slicer_synth.cxx
  )

##-----------------------------------------------------------------------------
##  Do we build a full plastimatch?
##-----------------------------------------------------------------------------
if (ITK_FOUND)
  if (GDCM_VERSION_2)
    message (WARNING "Plastimatch is built with GDCM 2.X -- DICOM-RT functions are disabled; if possible, please use GDCM 1.X instead.")
  endif ()
  set (FULL_PLASTIMATCH_BUILD 1)
endif ()

##-----------------------------------------------------------------------------
##  Library targets
##-----------------------------------------------------------------------------
if (F2C_LIBRARY AND UNIX)
  plm_add_library (f2c_helper "${F2C_HELPER_LIBRARY_SRC}" "" "")
endif ()

if (FULL_PLASTIMATCH_BUILD)
  # Despite claims to the contrary, you are not allowed to have a library
  # with the same name as the executable.  What happens is the executable 
  # tries to create a temporary library with the same name.  [MSVC 2005]
  # That is why the library is called plastimatch1 instead of plastimatch.
  set (PLASTIMATCH1_LIBRARY_DEPENDENCIES 
    ${ITK_LIBRARIES} ${GPUIT_LIBRARIES})

  # Add dcmtk libraries, if that is what the user configured
  if (DCMTK_FOUND AND NOT DCMTK_VERSION_STRING VERSION_LESS 3.6.0)
    set (PLASTIMATCH1_LIBRARY_DEPENDENCIES 
      ${PLASTIMATCH1_LIBRARY_DEPENDENCIES} ${DCMTK_LIBRARIES})
  endif ()

  # Ugh.  ITK is stupid.  They quietly dropped support for cygwin. 
  # However, you can still get 3.20.0 to work.  You need to run 
  # "make ITKFEM" after running "make" to workaround ITK's broken CMake 
  # script:
  #  http://old.nabble.com/compile-error-using-cygwin-td30879187.html
  # See also:
  #  http://public.kitware.com/Bug/view.php?id=11659
  if (CYGWIN)
    set (PLASTIMATCH1_LIBRARY_DEPENDENCIES 
      ${PLASTIMATCH1_LIBRARY_DEPENDENCIES} gdi32)
  endif ()

#  plm_add_library (
#    plastimatch1 
#    "${PLASTIMATCH1_LIBRARY_SRC}"
#    "${PLASTIMATCH1_LIBRARY_DEPENDENCIES}"
#    "")
endif ()

##-----------------------------------------------------------------------------
##  Executable targets
##-----------------------------------------------------------------------------
if (FULL_PLASTIMATCH_BUILD)
  message (STATUS "PLASTIMATCH_LIBS: ${PLASTIMATCH_LIBS}")
endif ()

# mondoshot requires WIN32, wx, dcmtk, sqlite3
if (WIN32 AND NOT CYGWIN AND wxWidgets_FOUND AND DCMTK_FOUND)
  add_executable (mondoshot WIN32 ${MONDOSHOT_SRC})
  target_link_libraries (mondoshot 
    ${wxWidgets_LIBRARIES} ${DCMTK_LIBRARIES} ${SQLITE_LIBRARIES})
  if (NOT BUILD_AGAINST_SLICER3)
    install (TARGETS mondoshot DESTINATION bin)
  endif ()
endif ()

##-----------------------------------------------------------------------------
##  Special targets for slicer
##-----------------------------------------------------------------------------
if (SLICER_FOUND)
  # The slicer plugin on windows can't find its dll easily.  Copy these 
  # over to the lib directory.  Also, copy over the plastimatch 
  # executable 
  if (WIN32)
    set (INDIR "${CMAKE_BINARY_DIR}/${CMAKE_CFG_INTDIR}")
  else ()
    set (INDIR "${CMAKE_BINARY_DIR}")
  endif ()
  if (SLICER_IS_SLICER3)
    set (PLUGIN_DIR "${CMAKE_BINARY_DIR}/${Slicer3_INSTALL_PLUGINS_BIN_DIR}")
  else ()
    set (PLUGIN_DIR "${CMAKE_BINARY_DIR}/${Slicer_CLIMODULES_BIN_DIR}")
    set (QTPLUGIN_DIR "${CMAKE_BINARY_DIR}/${Slicer_QTLOADABLEMODULES_BIN_DIR}")
  endif ()
  set (OUTDIR "${PLUGIN_DIR}/${CMAKE_CFG_INTDIR}")
  set (QTOUTDIR "${QTPLUGIN_DIR}/${CMAKE_CFG_INTDIR}")

  # Copy plastimatch executable
  plm_add_target_copy (plastimatch_slicer_copy_plastimatch_exe
    "${INDIR}/plastimatch${CMAKE_EXECUTABLE_SUFFIX}"
    "${OUTDIR}/plastimatch${CMAKE_EXECUTABLE_SUFFIX}"
    plastimatch
    )

  # Copy dlls
  if (WIN32)
    macro (plm_slicer_copy_dll TARGET SRC DEST QTDEST DEPENDENCY)
      plm_add_target_copy ("${TARGET}" "${SRC}" "${DEST}" "${DEPENDENCY}")
      if (SLICER_IS_SLICER4)
	set (QTTARGET "${TARGET}_qt")
	plm_add_target_copy ("${QTTARGET}" "${SRC}" "${QTDEST}" "${DEPENDENCY}")
      endif ()
    endmacro ()

#    plm_slicer_copy_dll (plastimatch_slicer_copy_plastimatch1
#      "${INDIR}/plastimatch1.dll"
#      "${OUTDIR}/plastimatch1.dll"
#      "${QTOUTDIR}/plastimatch1.dll"
#      plastimatch1
#      )
    plm_slicer_copy_dll (plastimatch_slicer_copy_gpuit
      "${INDIR}/gpuit.dll"
      "${OUTDIR}/gpuit.dll"
      "${QTOUTDIR}/gpuit.dll"
      gpuit
      )
    if (CUDA_FOUND)
      plm_slicer_copy_dll (plastimatch_slicer_copy_plmcuda
	"${INDIR}/plmcuda.dll"
	"${OUTDIR}/plmcuda.dll"
	"${QTOUTDIR}/plmcuda.dll"
	plmcuda
	)
    endif ()
    if (OPENCL_FOUND)
      plm_slicer_copy_dll (plastimatch_slicer_copy_plmopencl
	"${INDIR}/plmopencl.dll"
	"${OUTDIR}/plmopencl.dll"
	"${QTOUTDIR}/plmopencl.dll"
	plmopencl
	)
    endif ()
  endif ()
endif ()

# plastimatch modules
add_subdirectory(base)
add_subdirectory(cli)
add_subdirectory(dose)
add_subdirectory(qt)
add_subdirectory(reconstruct)
add_subdirectory(register)
add_subdirectory(segment)
add_subdirectory(standalone)
add_subdirectory(script)
add_subdirectory(sys)
add_subdirectory(test)
add_subdirectory(util)
