##-----------------------------------------------------------------------------
##  See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
##-----------------------------------------------------------------------------
project (src_plastimatch_sys)

configure_file (
    ${CMAKE_CURRENT_SOURCE_DIR}/plmsys_config.h.in
    ${PLM_BUILD_ROOT}/plmsys_config.h
)

include_directories (BEFORE ${CMAKE_CURRENT_SOURCE_DIR})

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
  print_and_exit.c
  pstring.h
  string_util.cxx string_util.h
  )

plm_add_library (
  plmsys
  "${PLMSYS_LIBRARY_SRC}" 
  "${PLMSYS_LIBRARY_DEPENDENCIES}"
  "")

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
