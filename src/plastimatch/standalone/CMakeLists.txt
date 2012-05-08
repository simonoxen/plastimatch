##-----------------------------------------------------------------------------
##  See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
##-----------------------------------------------------------------------------
project (src_plastimatch_standalone)

include_directories (BEFORE ${CMAKE_CURRENT_SOURCE_DIR})

##-----------------------------------------------------------------------------
##  CMAKE FLAGS
##-----------------------------------------------------------------------------
option (PLM_BUILD_SIFTND "Build siftnD program" ON)

##-----------------------------------------------------------------------------
##  SOURCE FILES
##-----------------------------------------------------------------------------
set (BRAGG_CURVE_SRC
  bragg_curve_main.c 
  bragg_curve_opts.c
  )
set (BSPLINE_SRC
  bspline_main.cxx 
  bspline_opts.cxx
  )
set (CHECK_GRAD_SRC
  check_grad.cxx
  check_grad_opts.cxx
  )
set (COMPUTE_DISTANCE_SRC
  compute_distance.cxx
  closest_point.cxx 
  read_obj.cxx
  )
set (COMPUTE_MEAN_IMAGE_SRC
  compute_mean_image_main.cxx
  )
set (CUDA_PROBE_SRC
  cuda_probe_main.c
  cuda_probe.h
  )
if (CUDA_FOUND)
  cuda_compile(CUDA_PROBE_WRAP cuda_probe.cu)
  set (CUDA_PROBE_SRC ${CUDA_PROBE_SRC} ${CUDA_PROBE_WRAP})
endif ()
set (DEMONS_SRC
  demons_main.cxx 
  demons_opts.cxx
  )
set (DICE_STATS_SRC
  dice_stats_main.cxx
  dice_statistics.cxx
  hausdorff_statistics.cxx
  )
set (DICOM_INFO_SRC
  dicom_info.cxx
  )
set (DICOM_UID_SRC
  dicom_uid_main.cxx
  )
set (DISTANCE_MAP_SRC
  distance_map.cxx
  )
set (DLIB_TRAIN_SRC 
  dlib_train.cxx
  )
set (DRR_SRC
  drr_main.cxx 
  drr_opts.cxx
  )
set (EXTRACT_CONTOUR_SRC
  extract_contour.cxx
  )
set (FDK_SRC
  fdk_main.cxx 
  fdk_opts.cxx
)
set (HND_TO_PFM_SRC
  hnd_to_pfm.cxx 
  )
set (LANDMARK_DIFF_SRC
  landmark_diff_main.cxx
  )
set (MERGE_VFS_SRC
  merge_vector_fields.cxx
  )
set (MEX_DRR_SRC
  mex_drr.c
  )
set (MHA_TO_RAW_SRC
  mha_to_raw.c
  )
set (MHA_TO_RTOG_DOSE_SRC
  mha_to_rtog_dose.cxx
  )
set (OPENCL_PROBE_SRC
  opencl_probe_main.c
  opencl_probe.cxx
  )
if (OPENCL_FOUND)
  plm_add_opencl_file (OPENCL_PROBE_SRC opencl_probe.cl)
endif ()
set (PROTON_DOSE_SRC
  proton_dose_main.cxx 
  proton_dose_opts.cxx
  )
set (RAW_TO_MHA_SRC
  raw_to_mha.c
  )
set (RTOG_TO_MHA_SRC
  rtog_to_mha.cxx
  )
set (SIFTnD_SRC
  siftnD.cxx
  )
set (UNION_MASK_SRC
  union_mask.cxx
  )
set (SHUFFLE_MHA_SRC
  shuffle_mha_main.cxx
  )
set (VF_INVERT_SRC
  vf_invert.cxx
  )
set (XF_TO_EMPIREFMT_SRC
  xf_to_empirefmt.cxx
  )

##-----------------------------------------------------------------------------
##  BUILD TARGETS
##-----------------------------------------------------------------------------
if (FULL_PLASTIMATCH_BUILD)
  plm_add_executable (compute_distance "${COMPUTE_DISTANCE_SRC}" 
    "${PLASTIMATCH_LIBS}" "${PLASTIMATCH_LDFLAGS}" 
    ${BUILD_IF_NOT_SLICER_EXT} ${INSTALL_IF_NOT_DEBIAN})
  plm_add_executable (compute_mean_image "${COMPUTE_MEAN_IMAGE_SRC}"
    "${PLASTIMATCH_LIBS}" "${PLASTIMATCH_LDFLAGS}" 
    ${BUILD_IF_NOT_SLICER_EXT} ${INSTALL_IF_NOT_DEBIAN})
  plm_add_executable (dice_stats "${DICE_STATS_SRC}"
    "${PLASTIMATCH_LIBS}" "${PLASTIMATCH_LDFLAGS}" 
    ${BUILD_IF_NOT_SLICER_EXT} ${INSTALL_IF_NOT_DEBIAN})
  plm_add_executable (dicom_info "${DICOM_INFO_SRC}"
    "${PLASTIMATCH_LIBS}" "${PLASTIMATCH_LDFLAGS}" 
    ${BUILD_IF_NOT_SLICER_EXT} ${INSTALL_IF_NOT_DEBIAN})
  plm_add_executable (distance_map "${DISTANCE_MAP_SRC}"
    "${PLASTIMATCH_LIBS}" "${PLASTIMATCH_LDFLAGS}" 
    ${BUILD_IF_NOT_SLICER_EXT} ${INSTALL_NEVER})
  plm_add_executable (extract_contour "${EXTRACT_CONTOUR_SRC}"
    "${PLASTIMATCH_LIBS}" "${PLASTIMATCH_LDFLAGS}" 
    ${BUILD_IF_NOT_SLICER_EXT} ${INSTALL_NEVER})
  plm_add_executable (landmark_diff "${LANDMARK_DIFF_SRC}" 
    "${PLASTIMATCH_LIBS}" "${PLASTIMATCH_LDFLAGS}" 
    ${BUILD_IF_NOT_SLICER_EXT} ${INSTALL_NEVER})
  plm_add_executable (merge_vfs  "${MERGE_VFS_SRC}"
    "${PLASTIMATCH_LIBS}" "${PLASTIMATCH_LDFLAGS}" 
    ${BUILD_IF_NOT_SLICER_EXT} ${INSTALL_IF_NOT_DEBIAN})
  plm_add_executable (mha_to_rtog_dose "${MHA_TO_RTOG_DOSE_SRC}"
    "${PLASTIMATCH_LIBS}" "${PLASTIMATCH_LDFLAGS}" 
    ${BUILD_IF_NOT_SLICER_EXT} ${INSTALL_NEVER})
  plm_add_executable (rtog_to_mha "${RTOG_TO_MHA_SRC}"
    "${PLASTIMATCH_LIBS}" "${PLASTIMATCH_LDFLAGS}" 
    ${BUILD_IF_NOT_SLICER_EXT} ${INSTALL_NEVER})
  plm_add_executable (shuffle_mha "${SHUFFLE_MHA_SRC}"
    "${PLASTIMATCH_LIBS}" "${PLASTIMATCH_LDFLAGS}" 
    ${BUILD_IF_NOT_SLICER_EXT} ${INSTALL_IF_NOT_DEBIAN})
  if (PLM_BUILD_SIFTND)
    plm_add_executable (siftnD "${SIFTnD_SRC}"
      "${PLASTIMATCH_LIBS}" "${PLASTIMATCH_LDFLAGS}" 
      ${BUILD_IF_NOT_SLICER_EXT} ${INSTALL_IF_NOT_DEBIAN})
  endif ()
  plm_add_executable (union_mask "${UNION_MASK_SRC}"
    "${PLASTIMATCH_LIBS}" "${PLASTIMATCH_LDFLAGS}" 
    ${BUILD_IF_NOT_SLICER_EXT} ${INSTALL_IF_NOT_DEBIAN})
  plm_add_executable (vf_invert "${VF_INVERT_SRC}"
    "${PLASTIMATCH_LIBS}" "${PLASTIMATCH_LDFLAGS}" 
    ${BUILD_IF_NOT_SLICER_EXT} ${INSTALL_IF_NOT_DEBIAN})
  plm_add_executable (xf_to_EMPIREFMT "${XF_TO_EMPIREFMT_SRC}"
    "${PLASTIMATCH_LIBS}" "${PLASTIMATCH_LDFLAGS}" 
    ${BUILD_IF_NOT_SLICER_EXT} ${INSTALL_NEVER})
endif ()


plm_add_executable (bspline "${BSPLINE_SRC}" 
  "${PLASTIMATCH_LIBS}" "${PLASTIMATCH_LDFLAGS}" 
  ${BUILD_IF_NOT_SLICER_EXT} ${INSTALL_NEVER})
plm_add_executable (check_grad "${CHECK_GRAD_SRC}" 
  "${PLASTIMATCH_LIBS}" "${PLASTIMATCH_LDFLAGS}" 
  ${BUILD_IF_NOT_SLICER_EXT} ${INSTALL_NEVER})
plm_add_executable (demons "${DEMONS_SRC}" 
  "${PLASTIMATCH_LIBS}" "${PLASTIMATCH_LDFLAGS}" 
  ${BUILD_IF_NOT_SLICER_EXT} ${INSTALL_NEVER})
plm_add_executable (dlib_train "${DLIB_TRAIN_SRC}" 
  "${PLASTIMATCH_LIBS}" "${PLASTIMATCH_LDFLAGS}" 
  ${BUILD_IF_NOT_SLICER_EXT} ${INSTALL_IF_NOT_DEBIAN})
plm_add_executable (drr "${DRR_SRC}" 
  "${PLASTIMATCH_LIBS}" "${PLASTIMATCH_LDFLAGS}" 
  ${BUILD_IF_NOT_SLICER_EXT} ${INSTALL_ALWAYS})
plm_add_executable (fdk "${FDK_SRC}" 
  "${PLASTIMATCH_LIBS}" "${PLASTIMATCH_LDFLAGS}" 
  ${BUILD_IF_NOT_SLICER_EXT} ${INSTALL_ALWAYS})
plm_add_executable (hnd_to_pfm "${HND_TO_PFM_SRC}" 
  "${PLASTIMATCH_LIBS}" "${PLASTIMATCH_LDFLAGS}" 
  ${BUILD_IF_NOT_SLICER_EXT} ${INSTALL_IF_NOT_DEBIAN})
plm_add_executable (mha_to_raw "${MHA_TO_RAW_SRC}"
  "" ""
  ${BUILD_IF_NOT_SLICER_EXT} ${INSTALL_IF_NOT_DEBIAN})
plm_add_executable (proton_dose "${PROTON_DOSE_SRC}" 
  "${PLASTIMATCH_LIBS}" "${PLASTIMATCH_LDFLAGS}" 
  ${BUILD_IF_NOT_SLICER_EXT} ${INSTALL_NEVER})
plm_add_executable (raw_to_mha "${RAW_TO_MHA_SRC}"
  "" "" 
  ${BUILD_IF_NOT_SLICER_EXT} ${INSTALL_IF_NOT_DEBIAN})

if (NOT BUILD_DEBIAN)
  plm_add_executable (bragg_curve "${BRAGG_CURVE_SRC}" 
    "${PLASTIMATCH_LIBS};${PLM_FORTRAN}" "${PLASTIMATCH_LDFLAGS}" 
    ${BUILD_IF_NOT_SLICER_EXT} ${INSTALL_NEVER})
endif ()

if (CUDA_FOUND)
  plm_add_executable (cuda_probe "${CUDA_PROBE_SRC}" 
    "${CUDA_LIBRARIES}" "" 
    ${BUILD_ALWAYS} ${INSTALL_IF_NOT_DEBIAN})
endif ()

if (MATLAB_FOUND)
  mex_target (mex_drr "${MEX_DRR_SRC}"
  "${PLASTIMATCH_LIBS}" "${PLASTIMATCH_LDFLAGS}")
endif ()

if (OPENCL_FOUND)
  plm_add_executable (opencl_probe "${OPENCL_PROBE_SRC}"
    "${PLASTIMATCH_LIBS}" "${PLASTIMATCH_LDFLAGS}" 
    ${BUILD_ALWAYS} ${INSTALL_IF_NOT_DEBIAN})
endif ()

# dicom_uid requires dcmtk
if (DCMTK_FOUND)
  plm_add_executable (dicom_uid "${DICOM_UID_SRC}" 
    "${PLASTIMATCH_LIBS}" "${PLASTIMATCH_LDFLAGS}" 
    ${BUILD_IF_NOT_SLICER_EXT} ${INSTALL_IF_NOT_DEBIAN})
endif ()
