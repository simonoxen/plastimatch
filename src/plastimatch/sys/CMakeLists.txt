##-----------------------------------------------------------------------------
##  See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
##-----------------------------------------------------------------------------
project (src_plastimatch_sys)

configure_file (
    ${CMAKE_CURRENT_SOURCE_DIR}/plmsys_config.h.in
    ${PLM_BUILD_ROOT}/plmsys_config.h
)

include_directories (BEFORE ${CMAKE_CURRENT_SOURCE_DIR})

## JAS 2012.06.22
##   Currently, it is necessary for appliations linking against
##   plmregister to also link against bstrlib

##-----------------------------------------------------------------------------
##  SOURCE FILES
##-----------------------------------------------------------------------------
set (PLMSYS_LIBRARY_SRC
  compiler_warnings.h
  delayload.c delayload.h
  dir_list.cxx dir_list.h
  file_util.cxx file_util.h
  logfile.cxx
  plm_endian.cxx
  plm_fwrite.cxx
  plm_int.h
  plm_path.h
  plm_timer.cxx plm_timer.h
  plm_version.h 
  print_and_exit.cxx
  pstring.h
  string_util.cxx string_util.h
  )

##-----------------------------------------------------------------------------
##  LIBRARY DEPENDENCIES
##-----------------------------------------------------------------------------
set (PLMSYS_LIBRARY_DEPENDENCIES 
  bstrlib
  )
if (LIBDL_FOUND)
  set (PLMSYS_LIBRARY_DEPENDENCIES
    ${PLMSYS_LIBRARY_DEPENDENCIES}
    dl
    )
endif ()

##-----------------------------------------------------------------------------
##  BUILD TARGETS
##-----------------------------------------------------------------------------
plm_add_library (
  plmsys
  "${PLMSYS_LIBRARY_SRC}" 
  "${PLMSYS_LIBRARY_DEPENDENCIES}"
  ""
  "${CMAKE_BINARY_DIR}/plmsys_config.h;plmsys.h;${CMAKE_BINARY_DIR}/plm_config.h")

##-----------------------------------------------------------------------------
##  SPECIAL RULES FOR SLICER
##-----------------------------------------------------------------------------
if (SLICER_FOUND)
  if (WIN32)
    plm_slicer_copy_dll (plastimatch_slicer_copy_plmsys
      "${INDIR}/plmsys.dll"
      "${OUTDIR}/plmsys.dll"
      "${QTOUTDIR}/plmsys.dll"
      plmsys
    )
  endif ()
endif ()
