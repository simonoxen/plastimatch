##-----------------------------------------------------------------------------
##  See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
##-----------------------------------------------------------------------------
project (src_plastimatch_standalone)

include_directories (BEFORE ${CMAKE_CURRENT_SOURCE_DIR})

set (BRAGG_CURVE_SRC
  bragg_curve_main.c 
  bragg_curve_opts.c bragg_curve_opts.h
  )
set (BSPLINE_SRC
  bspline_main.cxx 
  bspline_opts.cxx bspline_opts.h
  )
set (CHECK_GRAD_SRC
  check_grad.cxx
  check_grad_opts.cxx check_grad_opts.h 
  )
set (COMPUTE_DISTANCE_SRC
  compute_distance.cxx compute_distance.h
  closest_point.cxx 
  read_obj.cxx
  )
set (COMPUTE_MEAN_IMAGE_SRC
  compute_mean_image_main.cxx
  )
set (DEMONS_SRC
  demons_main.cxx 
  demons_opts.cxx demons_opts.h
  )
set (DICE_STATS_SRC
  dice_stats_main.cxx
  dice_statistics.cxx dice_statistics.h
  hausdorff_statistics.cxx hausdorff_statistics.h
  )
set (DICOM_INFO_SRC
  dicom_info.cxx
  )
#set (DICOM_UID_SRC
#  dicom_uid_main.cxx
#  dcmtk_uid.cxx
#  dcmtk_uid.h
#  )
set (DISTANCE_MAP_SRC
  distance_map.cxx
  )
set (DLIB_TRAIN_SRC 
  dlib_train.cxx
  )
set (DRR_SRC
  drr_main.cxx 
  drr_opts.cxx drr_opts.h
  )
set (FDK_SRC
  fdk_main.cxx 
  fdk_opts.cxx fdk_opts.h
)
set (HND_TO_PFM_SRC
  hnd_to_pfm.cxx 
  )
set (MHA_TO_RAW_SRC
  mha_to_raw.c
  )
set (PROTON_DOSE_SRC
  proton_dose_main.cxx 
  proton_dose_opts.cxx proton_dose_opts.h
  )
set (RAW_TO_MHA_SRC
  raw_to_mha.c
  )


if (FULL_PLASTIMATCH_BUILD)
  plm_add_executable (compute_distance "${COMPUTE_DISTANCE_SRC}" 
    "${PLASTIMATCH_LIBS}" "${GPUIT_EXE_LDFLAGS}" 
    ${BUILD_IF_NOT_SLICER_EXT} ${INSTALL_IF_NOT_DEBIAN})
  plm_add_executable (compute_mean_image "${COMPUTE_MEAN_IMAGE_SRC}"
    "${PLASTIMATCH_LIBS}" "${GPUIT_EXE_LDFLAGS}" 
    ${BUILD_IF_NOT_SLICER_EXT} ${INSTALL_IF_NOT_DEBIAN})
  plm_add_executable (dice_stats "${DICE_STATS_SRC}"
    "${PLASTIMATCH_LIBS}" "${GPUIT_EXE_LDFLAGS}" 
    ${BUILD_IF_NOT_SLICER_EXT} ${INSTALL_IF_NOT_DEBIAN})
  plm_add_executable (dicom_info "${DICOM_INFO_SRC}"
    "${PLASTIMATCH_LIBS}" "${GPUIT_EXE_LDFLAGS}" 
    ${BUILD_IF_NOT_SLICER_EXT} ${INSTALL_IF_NOT_DEBIAN})
  plm_add_executable (distance_map "${DISTANCE_MAP_SRC}"
    "${PLASTIMATCH_LIBS}" "${GPUIT_EXE_LDFLAGS}" 
    ${BUILD_IF_NOT_SLICER_EXT} ${INSTALL_NEVER})
endif ()


plm_add_executable (bspline "${BSPLINE_SRC}" 
  "${GPUIT_LIBRARIES}" "${PLMCUDA_EXE_LDFLAGS}" 
  ${BUILD_IF_NOT_SLICER_EXT} ${INSTALL_NEVER})
plm_add_executable (check_grad "${CHECK_GRAD_SRC}" 
  "${GPUIT_LIBRARIES}" "${GPUIT_EXE_LDFLAGS}" 
  ${BUILD_IF_NOT_SLICER_EXT} ${INSTALL_NEVER})
plm_add_executable (demons "${DEMONS_SRC}" 
  "${GPUIT_LIBRARIES}" "${PLMCUDA_EXE_LDFLAGS}" 
  ${BUILD_IF_NOT_SLICER_EXT} ${INSTALL_NEVER})
plm_add_executable (dlib_train "${DLIB_TRAIN_SRC}" 
  "${GPUIT_LIBRARIES}" "${PLMCUDA_EXE_LDFLAGS}" 
  ${BUILD_IF_NOT_SLICER_EXT} ${INSTALL_IF_NOT_DEBIAN})
plm_add_executable (drr "${DRR_SRC}" 
  "${GPUIT_LIBRARIES}" "${PLMCUDA_EXE_LDFLAGS}" 
  ${BUILD_IF_NOT_SLICER_EXT} ${INSTALL_ALWAYS})
plm_add_executable (fdk "${FDK_SRC}" 
  "${GPUIT_LIBRARIES}" "${PLMCUDA_EXE_LDFLAGS}" 
  ${BUILD_IF_NOT_SLICER_EXT} ${INSTALL_ALWAYS})
plm_add_executable (hnd_to_pfm "${HND_TO_PFM_SRC}" 
  "${GPUIT_LIBRARIES}" "${GPUIT_EXE_LDFLAGS}" 
  ${BUILD_IF_NOT_SLICER_EXT} ${INSTALL_IF_NOT_DEBIAN})
plm_add_executable (mha_to_raw "${MHA_TO_RAW_SRC}"
  "" ""
  ${BUILD_IF_NOT_SLICER_EXT} ${INSTALL_IF_NOT_DEBIAN})
plm_add_executable (proton_dose "${PROTON_DOSE_SRC}" 
  "${GPUIT_LIBRARIES}" "${GPUIT_EXE_LDFLAGS}" 
  ${BUILD_IF_NOT_SLICER_EXT} ${INSTALL_NEVER})
plm_add_executable (raw_to_mha "${RAW_TO_MHA_SRC}"
  "" "" 
  ${BUILD_IF_NOT_SLICER_EXT} ${INSTALL_IF_NOT_DEBIAN})


if (NOT BUILD_DEBIAN)
  plm_add_executable (bragg_curve "${BRAGG_CURVE_SRC}" 
    "${GPUIT_LIBRARIES}" "${GPUIT_EXE_LDFLAGS}" 
    ${BUILD_IF_NOT_SLICER_EXT} ${INSTALL_NEVER})
endif ()


# dicom_uid requires dcmtk
#if (DCMTK_FOUND)
#  plm_add_executable (dicom_uid "${DICOM_UID_SRC}" 
#    "${DCMTK_LIBRARIES}" "" 
#    ${BUILD_IF_NOT_SLICER_EXT} ${INSTALL_IF_NOT_DEBIAN})
#endif ()
