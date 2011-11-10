##-----------------------------------------------------------------------------
##  Welcome to the Plastimatch CMakeLists.txt file
##-----------------------------------------------------------------------------
##  See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
##-----------------------------------------------------------------------------
project (plastimatch_slicer)

##-----------------------------------------------------------------------------
##  This directory only contains things that depend on 3D Slicer
##-----------------------------------------------------------------------------
if (NOT SLICER_FOUND)
  return ()
endif ()

##-----------------------------------------------------------------------------
##  Include directories
##-----------------------------------------------------------------------------
include_directories (BEFORE ${src_plastimatch_SOURCE_DIR})
include_directories (BEFORE ${CMAKE_CURRENT_SOURCE_DIR})
include_directories (BEFORE ${CMAKE_BINARY_DIR})
include_directories (BEFORE ${CMAKE_CURRENT_BINARY_DIR})
include_directories (AFTER ${BSTRLIB_INCLUDE_DIR})
include_directories (AFTER ${LIBLBFGS_INCLUDE_DIR})
include_directories (AFTER ${MSINTTYPES_INCLUDE_DIR})
include_directories (AFTER ${SQLITE_INCLUDE_DIR})
include_directories (AFTER ${SLICER_INCLUDE_DIRS}
  #${Slicer_Libs_INCLUDE_DIRS}
  #${Slicer_Base_INCLUDE_DIRS}
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
set (PLASTIMATCH_SLICER_DVH_SRC
  plastimatch_slicer_dvh.cxx
  )
set (PLASTIMATCH_SLICER_LANDWARP_SRC
  plastimatch_slicer_landwarp.cxx
  )
set (PLASTIMATCH_SLICER_SYNTH_SRC
  plastimatch_slicer_synth.cxx
  )
set (PLASTIMATCH_SLICER_XFORMWARP_SRC
  plastimatch_slicer_xformwarp.cxx
  )
set (PLASTIMATCH_SLICER_THRESHBOX_SRC
  plastimatch_slicer_threshbox.cxx
  )

##-----------------------------------------------------------------------------
##  Executable targets
##-----------------------------------------------------------------------------
plm_add_slicer_executable (plastimatch_slicer_bspline
  "${PLASTIMATCH_SLICER_BSPLINE_SRC}" "${PLASTIMATCH_LIBS}" 
  "${GPUIT_EXE_LDFLAGS}")
plm_add_slicer_executable (plastimatch_slicer_dicomrt_import
  "${PLASTIMATCH_SLICER_DICOMRT_IMPORT_SRC}" "${PLASTIMATCH_LIBS}" 
  "${GPUIT_EXE_LDFLAGS}")
plm_add_slicer_executable (plastimatch_slicer_dicomrt_export
  "${PLASTIMATCH_SLICER_DICOMRT_EXPORT_SRC}" "${PLASTIMATCH_LIBS}" 
  "${GPUIT_EXE_LDFLAGS}")
plm_add_slicer_executable (plastimatch_slicer_dvh
  "${PLASTIMATCH_SLICER_DVH_SRC}" "${PLASTIMATCH_LIBS}" 
  "${GPUIT_EXE_LDFLAGS}")
plm_add_slicer_executable (plastimatch_slicer_landwarp 
  "${PLASTIMATCH_SLICER_LANDWARP_SRC}" "${PLASTIMATCH_LIBS}" 
  "${GPUIT_EXE_LDFLAGS}")
plm_add_slicer_executable (plastimatch_slicer_synth
  "${PLASTIMATCH_SLICER_SYNTH_SRC}" "${PLASTIMATCH_LIBS}" 
  "${GPUIT_EXE_LDFLAGS}")
plm_add_slicer_executable (plastimatch_slicer_xformwarp
  "${PLASTIMATCH_SLICER_XFORMWARP_SRC}" "${PLASTIMATCH_LIBS}" 
  "${GPUIT_EXE_LDFLAGS}")
plm_add_slicer_executable (plastimatch_slicer_threshbox
  "${PLASTIMATCH_SLICER_THRESHBOX_SRC}" "${PLASTIMATCH_LIBS}" 
  "${GPUIT_EXE_LDFLAGS}")

##-----------------------------------------------------------------------------
##  Loadable modules
##-----------------------------------------------------------------------------
if (SLICER_IS_SLICER4)
  add_subdirectory (PlastimatchDVH_Slicer4Plugin)
endif ()
