##-----------------------------------------------------------------------------
##  Welcome to the Plastimatch CMakeLists.txt file
##-----------------------------------------------------------------------------
##  See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
##-----------------------------------------------------------------------------
PROJECT (plastimatch_slicer)

##-----------------------------------------------------------------------------
##  This directory only contains things that depend on 3D Slicer
##-----------------------------------------------------------------------------
if (NOT SLICER_FOUND)
  return ()
endif ()

##-----------------------------------------------------------------------------
##  Include directories
##-----------------------------------------------------------------------------
INCLUDE_DIRECTORIES (BEFORE ${src_plastimatch_SOURCE_DIR})
INCLUDE_DIRECTORIES (BEFORE ${CMAKE_CURRENT_SOURCE_DIR})
INCLUDE_DIRECTORIES (BEFORE ${CMAKE_BINARY_DIR})
INCLUDE_DIRECTORIES (BEFORE ${CMAKE_CURRENT_BINARY_DIR})
INCLUDE_DIRECTORIES (AFTER ${BSTRLIB_INCLUDE_DIR})
INCLUDE_DIRECTORIES (AFTER ${LIBLBFGS_INCLUDE_DIR})
INCLUDE_DIRECTORIES (AFTER ${MSINTTYPES_INCLUDE_DIR})
INCLUDE_DIRECTORIES (AFTER ${SQLITE_INCLUDE_DIR})
INCLUDE_DIRECTORIES (AFTER ${SLICER_INCLUDE_DIRS}
  #${Slicer_Libs_INCLUDE_DIRS}
  #${Slicer_Base_INCLUDE_DIRS}
  )

##-----------------------------------------------------------------------------
##  SOURCE FILES (EXECUTABLES)
##-----------------------------------------------------------------------------
SET (PLASTIMATCH_SLICER_BSPLINE_SRC
  plastimatch_slicer_bspline.cxx
  )
SET (PLASTIMATCH_SLICER_DICOMRT_IMPORT_SRC
  plastimatch_slicer_dicomrt_import.cxx
  )
SET (PLASTIMATCH_SLICER_DICOMRT_EXPORT_SRC
  plastimatch_slicer_dicomrt_export.cxx
  )
SET (PLASTIMATCH_SLICER_LANDWARP_SRC
  plastimatch_slicer_landwarp.cxx
  )
SET (PLASTIMATCH_SLICER_SYNTH_SRC
  plastimatch_slicer_synth.cxx
  )

##-----------------------------------------------------------------------------
##  Executable targets
##-----------------------------------------------------------------------------
IF (SLICER_FOUND)
  PLM_ADD_SLICER_EXECUTABLE (plastimatch_slicer_bspline
    "${PLASTIMATCH_SLICER_BSPLINE_SRC}" "${PLASTIMATCH_LIBS}" 
    "${GPUIT_EXE_LDFLAGS}")
  PLM_ADD_SLICER_EXECUTABLE (plastimatch_slicer_dicomrt_import
    "${PLASTIMATCH_SLICER_DICOMRT_IMPORT_SRC}" "${PLASTIMATCH_LIBS}" 
    "${GPUIT_EXE_LDFLAGS}")
  PLM_ADD_SLICER_EXECUTABLE (plastimatch_slicer_dicomrt_export
    "${PLASTIMATCH_SLICER_DICOMRT_EXPORT_SRC}" "${PLASTIMATCH_LIBS}" 
    "${GPUIT_EXE_LDFLAGS}")
  PLM_ADD_SLICER_EXECUTABLE (plastimatch-slicer-landwarp 
    "${PLASTIMATCH_SLICER_LANDWARP_SRC}" "${PLASTIMATCH_LIBS}" 
    "${GPUIT_EXE_LDFLAGS}")
  PLM_ADD_SLICER_EXECUTABLE (plastimatch_slicer_synth
    "${PLASTIMATCH_SLICER_SYNTH_SRC}" "${PLASTIMATCH_LIBS}" 
    "${GPUIT_EXE_LDFLAGS}")
ENDIF ()
