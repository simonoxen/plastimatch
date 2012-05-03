##-----------------------------------------------------------------------------
##  See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
##-----------------------------------------------------------------------------
project (src_plastimatch_cli)

configure_file (
    ${CMAKE_CURRENT_SOURCE_DIR}/plmcli_config.h.in
    ${PLM_BUILD_ROOT}/plmcli_config.h
)

include_directories (BEFORE ${CMAKE_CURRENT_SOURCE_DIR})

##-----------------------------------------------------------------------------
##  SOURCE FILES
##-----------------------------------------------------------------------------
set (PLMCLP_LIBRARY_SRC
  plm_clp.cxx plm_clp.h
  )

set (PLASTIMATCH_SRC
  pcmd_add.cxx pcmd_add.h
  pcmd_adjust.cxx pcmd_adjust.h
  pcmd_autolabel_train.cxx pcmd_autolabel_train.h
  pcmd_autolabel.cxx pcmd_autolabel.h
  pcmd_compare.cxx pcmd_compare.h
  pcmd_compose.cxx pcmd_compose.h
  pcmd_crop.cxx pcmd_crop.h
  pcmd_diff.cxx pcmd_diff.h
  pcmd_drr.cxx pcmd_drr.h 
  pcmd_dvh.cxx pcmd_dvh.h
  pcmd_header.cxx pcmd_header.h
  pcmd_mask.cxx pcmd_mask.h
  pcmd_probe.cxx pcmd_probe.h
  pcmd_resample.cxx pcmd_resample.h
  pcmd_script.cxx pcmd_script.h
  pcmd_segment.cxx pcmd_segment.h
  pcmd_stats.cxx pcmd_stats.h
  pcmd_synth.cxx pcmd_synth.h
  pcmd_synth_vf.cxx pcmd_synth_vf.h
  pcmd_thumbnail.cxx pcmd_thumbnail.h 
  pcmd_warp.cxx pcmd_warp.h 
  pcmd_warp_dij.cxx 
  pcmd_warp_pointset.cxx 
  pcmd_xio_dvh.cxx pcmd_xio_dvh.h
  pcmd_xf_convert.cxx pcmd_xf_convert.h 
  plastimatch_main.cxx
  )

set (LANDMARK_WARP_SRC
  landmark_warp_main.cxx
  )

set (PLM_CLI_LIBRARIES ${PLASTIMATCH_LIBS} plmclp)

##-----------------------------------------------------------------------------
##  BUILD TARGETS
##-----------------------------------------------------------------------------
plm_add_library (
  plmclp
  "${PLMCLP_LIBRARY_SRC}" 
  ""
  "")
plm_add_executable (plastimatch "${PLASTIMATCH_SRC}"
  "${PLM_CLI_LIBRARIES}" "${PLASTIMATCH_LDFLAGS}" 
  ${BUILD_ALWAYS} ${INSTALL_ALWAYS})
plm_add_executable (landmark_warp "${LANDMARK_WARP_SRC}" 
  "${PLM_CLI_LIBRARIES}" "${PLASTIMATCH_LDFLAGS}" 
  ${BUILD_IF_NOT_SLICER_EXT} ${INSTALL_ALWAYS})
