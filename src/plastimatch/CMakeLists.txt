##-----------------------------------------------------------------------------
##  Welcome to the Plastimatch CMakeLists.txt file
##-----------------------------------------------------------------------------
##  See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
##-----------------------------------------------------------------------------
PROJECT (src_plastimatch)

##-----------------------------------------------------------------------------
##  Include directories
##-----------------------------------------------------------------------------
INCLUDE_DIRECTORIES (BEFORE ${CMAKE_CURRENT_SOURCE_DIR})
INCLUDE_DIRECTORIES (BEFORE ${CMAKE_BINARY_DIR})
INCLUDE_DIRECTORIES (BEFORE ${CMAKE_CURRENT_BINARY_DIR})
INCLUDE_DIRECTORIES (AFTER ${BSTRLIB_INCLUDE_DIR})
INCLUDE_DIRECTORIES (AFTER ${LIBLBFGS_INCLUDE_DIR})
INCLUDE_DIRECTORIES (AFTER ${MSINTTYPES_INCLUDE_DIR})
INCLUDE_DIRECTORIES (AFTER ${SQLITE_INCLUDE_DIR})

# Don't do the below.  A user-built f2c directory contains ctype.h, which 
# conflicts with the C/C++ header of the same name.
#IF (PLM_USE_F2C)
#  INCLUDE_DIRECTORIES (BEFORE ${F2C_INCLUDE_DIR})
#ENDIF ()

IF (BROOK_FOUND)
  INCLUDE_DIRECTORIES (AFTER ${BROOK_INCLUDE_DIR})
ENDIF ()

IF (CUDA_FOUND)
  INCLUDE_DIRECTORIES (AFTER ${CUDA_INCLUDE_DIRS})
  #INCLUDE_DIRECTORIES (AFTER ${CUDA_CUT_INCLUDE_DIR})
ENDIF ()

IF (DCMTK_FOUND)
  INCLUDE_DIRECTORIES (AFTER ${DCMTK_INCLUDE_DIR})
ENDIF ()

INCLUDE_DIRECTORIES (AFTER ${DLIB_INCLUDE_DIR})

IF (FFTW_FOUND)
  INCLUDE_DIRECTORIES (BEFORE ${FFTW_INCLUDE_DIR})
  LINK_DIRECTORIES (${FFTW_DIR})
ENDIF ()

IF (NOT HAVE_GETOPT_LONG)
  INCLUDE_DIRECTORIES (AFTER "${CMAKE_SOURCE_DIR}/libs/getopt")
ENDIF ()

IF (OPENCL_FOUND)
  INCLUDE_DIRECTORIES (BEFORE ${OPENCL_INCLUDE_DIR})
ENDIF ()

IF (MATLAB_FOUND)
  INCLUDE_DIRECTORIES (AFTER ${MATLAB_INCLUDE_DIRS})
ENDIF ()

IF (PANTHEIOS_FOUND)
  INCLUDE_DIRECTORIES (AFTER ${STLSOFT_INCLUDE_DIR})
  INCLUDE_DIRECTORIES (AFTER ${PANTHEIOS_INCLUDE_DIR})
ENDIF ()

IF (QT4_FOUND AND QT_QTSQL_FOUND)
  INCLUDE_DIRECTORIES (AFTER ${QT_QTSQL_INCLUDE_DIR})
ENDIF ()

INCLUDE_DIRECTORIES (AFTER ${RANSAC_INCLUDE_DIRS})

IF (wxWidgets_FOUND)
  INCLUDE_DIRECTORIES (BEFORE ${wxWidgets_INCLUDE_DIRS})
ENDIF ()

##-----------------------------------------------------------------------------
##  BROOK FILES
##-----------------------------------------------------------------------------
IF (BROOK_FOUND)
  BRCC_FILE(fdk_brook_kernel.br)
  SET_SOURCE_FILES_PROPERTIES(fdk_brook.cpp PROPERTIES
    OBJECT_DEPENDS "fdk_brook_kernel.cpp")

  BRCC_FILE(convolution_brook_kernel.br)
  BRCC_FILE(demons_brook_kernel.br)
  SET_SOURCE_FILES_PROPERTIES(demons_brook_kernel_wrap.cpp PROPERTIES
    OBJECT_DEPENDS "demons_brook_kernel.cpp")
  SET_SOURCE_FILES_PROPERTIES(convolution_brook_kernel_wrap.cpp PROPERTIES
    OBJECT_DEPENDS "convolution_brook_kernel.cpp")

  BRCC_FILE(toy_kernels.br)
  SET_SOURCE_FILES_PROPERTIES(toy_kernels.cpp PROPERTIES
    OBJECT_DEPENDS "toy_kernels.cpp")
ENDIF ()

##-----------------------------------------------------------------------------
##  PLMSYS/GPUIT/PLASTIMATCH LIBRARY SOURCE FILES
##-----------------------------------------------------------------------------
SET (PLMSYS_LIBRARY_SRC
  file_util.cxx file_util.h
  logfile.c logfile.h
  interpolate.c interpolate.h
  plm_timer.c plm_timer.h
  print_and_exit.c print_and_exit.h 
  )

SET (GPUIT_LIBRARY_SRC
  bowtie_correction.cxx bowtie_correction.h
  bstring_util.cxx bstring_util.h
  bspline.cxx bspline.h
  bspline_mi.cxx bspline_mi.h
  bspline_mse.cxx bspline_mse.h
  bspline_landmarks.cxx bspline_landmarks.h
  bspline_optimize.cxx bspline_optimize.h
  bspline_optimize_liblbfgs.cxx bspline_optimize_liblbfgs.h
  bspline_optimize_lbfgsb.cxx bspline_optimize_lbfgsb.h
  bspline_optimize_nlopt.cxx bspline_optimize_nlopt.h
  bspline_optimize_steepest.cxx bspline_optimize_steepest.h
  bspline_opts.cxx bspline_opts.h
  bspline_regularize.cxx bspline_regularize.h
  bspline_warp.cxx bspline_warp.h
  bspline_xform.cxx bspline_xform.h
  delayload.c delayload.h
  reg.cxx reg.h
  demons.cxx demons.h
  demons_cpu.cxx
  demons_misc.cxx demons_misc.h
  dir_list.cxx dir_list.h
  drr.cxx drr.h
  drr_trilin.cxx drr_trilin.h
  fdk.cxx fdk.h
  fwrite_block.cxx fwrite_block.h
  hnd_io.cxx hnd_io.h
  math_util.h
  mha_io.cxx mha_io.h
  rpl_volume.cxx rpl_volume.h
  pointset.cxx pointset.h 
  proj_image.cxx proj_image.h
  proj_image_dir.cxx proj_image_dir.h
  proj_matrix.cxx proj_matrix.h 
  proton_dose.cxx proton_dose.h 
  ray_trace_exact.cxx ray_trace_exact.h 
  ray_trace_uniform.cxx ray_trace_uniform.h 
  string_util.cxx string_util.h
  threading.cxx threading.h
  vf.cxx vf.h
  vf_convolve.cxx vf_convolve.h 
  vf_stats.cxx vf_stats.h 
  volume.cxx volume.h
  volume_header.cxx volume_header.h 
  volume_limit.cxx volume_limit.h
  volume_macros.h
  xpm.cxx xpm.h
  )

IF (BROOK_FOUND)
  SET (GPUIT_LIBRARY_SRC
    ${GPUIT_LIBRARY_SRC}
    demons_brook.cpp
    demons_brook_kernel_wrap.cpp
    demons_brook_kernel.br
    demons_brook_kernel.h
    convolution_brook_kernel.br
    convolution_brook_kernel_wrap.cpp
    convolution_brook_kernel.h
    )
ENDIF ()

IF (CUDA_FOUND)
  SET (PLMCUDA_LIBRARY_SRC
    bspline_cuda.cpp bspline_cuda.cu bspline_cuda.h
    cuda_kernel_util.cu cuda_kernel_util.h
    cuda_mem.cu cuda_mem.h
    cuda_util.cu cuda_util.h
    demons_cuda.cu demons_cuda.h
    demons_misc.cxx demons_misc.h
    drr_cuda.cpp drr_cuda.cu drr_cuda.h drr_cuda_p.h 
    fdk_cuda.cpp fdk_cuda.cu fdk_cuda_p.h fdk_cuda.h
    )

  IF (BUILD_SHARED_LIBS)      # << Dynamic loading
    SET (PLMCUDA_LIBRARY_SRC
      ${PLMCUDA_LIBRARY_SRC}
      volume.cxx          # for demons_cuda.cu
      volume.h            # for demons_cuda.cu
    )
    SET (PLMCUDA_LIBRARY_DEPENDENCIES
      plmsys
    )
    CUDA_ADD_LIBRARY (plmcuda SHARED
      ${PLMCUDA_LIBRARY_SRC}
      )
    SET_TARGET_PROPERTIES (plmcuda PROPERTIES 
      ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
      LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
      RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
    TARGET_LINK_LIBRARIES (plmcuda
      ${PLMCUDA_LIBRARY_DEPENDENCIES}
      )
    INSTALL (TARGETS plmcuda
      RUNTIME DESTINATION bin
      LIBRARY DESTINATION lib
      )
  ELSE ()        # << Static linking
    SET (GPUIT_LIBRARY_SRC
      ${GPUIT_LIBRARY_SRC}
      ${PLMCUDA_LIBRARY_SRC}
      )
    set (PLM_CUDA_FILES
      bspline_cuda.cu 
      cuda_util.cu
      cuda_kernel_util.cu
      cuda_mem.cu
      demons_cuda.cu
      drr_cuda.cu
      fdk_cuda.cu
      )
    CUDA_COMPILE (CUDA_WRAPPERS ${PLM_CUDA_FILES})
    SET (GPUIT_LIBRARY_SRC
      ${GPUIT_LIBRARY_SRC}
      ${CUDA_WRAPPERS}
      ${PLM_CUDA_FILES}
      )
  ENDIF ()
ENDIF ()

IF (FFTW_FOUND)
  SET (GPUIT_LIBRARY_SRC
    ${GPUIT_LIBRARY_SRC}
    ramp_filter.c ramp_filter.h
   )
ENDIF ()

IF (OPENCL_FOUND)
  # Define OpenCL Plugin (plmopencl)
  SET (PLMOPENCL_LIBRARY_SRC
    autotune_opencl.cxx autotune_opencl.h
    opencl_probe.cxx opencl_probe.h opencl_probe.cl
    opencl_util.cxx opencl_util.h
    opencl_util_nvidia.cxx opencl_util_nvidia.h
    )

  IF (BUILD_SHARED_LIBS)      # << Dynamic loading

    SET (PLMOPENCL_LIBRARY_SRC
      ${PLMOPENCL_LIBRARY_SRC}
    )

    SET (PLMOPENCL_LIBRARY_DEPENDENCIES
      plmsys
    )

    # OpenCL Plugin is ALWAYS shared!
    ADD_LIBRARY (plmopencl SHARED
      ${PLMOPENCL_LIBRARY_SRC}
      )
    SET_TARGET_PROPERTIES (plmopencl PROPERTIES 
      ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
      LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
      RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")

    # Be sure to link against libOpenCL and other depends
    TARGET_LINK_LIBRARIES (plmopencl
      ${OPENCL_LIBRARIES}
      ${PLMOPENCL_LIBRARY_DEPENDENCIES}
      )

    INSTALL (TARGETS plmopencl
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION lib
    )

  ELSE ()        # << Static linking

    SET (GPUIT_LIBRARY_SRC
      ${GPUIT_LIBRARY_SRC}
      ${PLMOPENCL_LIBRARY_SRC}
    )

  ENDIF ()

  SET (GPUIT_LIBRARY_SRC
    ${GPUIT_LIBRARY_SRC}
    demons_opencl.cxx
    demons_opencl_p.h
    demons_opencl.cl
    drr_opencl.cxx
    drr_opencl_p.h
    drr_opencl.h
    drr_opencl.cl
    fdk_opencl.cxx 
    fdk_opencl_p.h
    fdk_opencl.h
    fdk_opencl.cl
    )
  PLM_ADD_OPENCL_FILE (GPUIT_LIBRARY_SRC demons_opencl.cl)
  PLM_ADD_OPENCL_FILE (GPUIT_LIBRARY_SRC drr_opencl.cl)
  PLM_ADD_OPENCL_FILE (GPUIT_LIBRARY_SRC fdk_opencl.cl)

ENDIF ()

IF (F2C_LIBRARY)
  SET (F2C_HELPER_LIBRARY_SRC
    main__.c
    )
ENDIF ()

SET (PLASTIMATCH_LIBRARY_SRC
  autolabel_trainer.cxx autolabel_trainer.h 
  autolabel_ransac_est.cxx autolabel_ransac_est.h 
  astroid_dose.cxx astroid_dose.h
  cxt_extract.cxx cxt_extract.h
  cxt_io.cxx cxt_io.h 
  dicom_util.cxx dicom_util.h
  direction_cosines.h
  gcs_metric.h gcs_metric.hxx 
  gpuit_bspline.cxx gpuit_bspline.h 
  gpuit_demons.cxx gpuit_demons.h 
  img_metadata.cxx img_metadata.h 
  itkAndConstantToImageFilter.h
  itk_crop.cxx itk_crop.h 
  itk_demons.cxx itk_demons.h 
  itk_dicom_load.cxx itk_dicom_load.h 
  itk_dicom_save.cxx itk_dicom_save.h 
  itk_image.cxx itk_image.h 
  itk_image_cast.cxx itk_image_cast.h 
  itk_image_load.h itk_image_load.txx
  itk_image_load_double.cxx
  itk_image_load_float.cxx
  itk_image_load_int32.cxx
  itk_image_load_short.cxx
  itk_image_load_uchar.cxx
  itk_image_load_uint32.cxx
  itk_image_load_ushort.cxx
  itk_image_load_vec.cxx
  itk_image_save.cxx itk_image_save.h 
  itk_image_stats.cxx itk_image_stats.h 
  itk_mask.cxx itk_mask.h 
  itk_pointset.cxx itk_pointset.h
  itk_optim.cxx itk_optim.h 
  itk_registration.cxx itk_registration.h 
  itk_tps.cxx itk_tps.h 
  itk_volume_header.cxx itk_volume_header.h
  itk_warp.cxx itk_warp.h 
  landmark_warp.cxx landmark_warp.h 
  mc_dose.cxx mc_dose.h
  plm_clp.h
  plm_drr_api.cxx plm_drr_api.h 
  plm_file_format.cxx plm_file_format.h 
  plm_HausdorffDistanceImageFilter.hxx plm_HausdorffDistanceImageFilter.h
  plm_DirectedHausdorffDistanceImageFilter.hxx plm_DirectedHausdorffDistanceImageFilter.h
  plm_image.cxx plm_image.h 
  plm_image_convert.cxx plm_image_convert.h 
  plm_image_header.cxx plm_image_header.h 
  plm_image_patient_position.cxx plm_image_patient_position.h 
  plm_image_type.cxx plm_image_type.h 
  plm_int.h
  plm_parms.cxx plm_parms.h 
  plm_register_loadable.cxx plm_register_loadable.h 
  plm_registration_api.cxx plm_registration_api.h 
  plm_stages.cxx plm_stages.h 
  plm_version.h 
  plm_warp.cxx plm_warp.h 
  rasterizer.cxx rasterizer.h
  rbf_gauss.cxx rbf_gauss.h
  rbf_wendland.cxx rbf_wendland.h
  rbf_cluster.cxx rbf_cluster.h
  referenced_dicom_dir.cxx referenced_dicom_dir.h 
  render_polyline.cxx render_polyline.h
  resample_mha.cxx resample_mha.h 
  rtds.cxx rtds.h
  rtds_dicom.cxx rtds_dicom.h
  rtds_warp.cxx rtds_warp.h
  rtss.cxx rtss.h
  rtss_polyline_set.cxx rtss_polyline_set.h 
  rtss_structure.cxx rtss_structure.h 
  segment_body.cxx segment_body.h
  simplify_points.cxx simplify_points.h
  slice_extract.cxx slice_extract.h
  ss_img_extract.cxx ss_img_extract.h
  ss_img_stats.cxx ss_img_stats.h 
  ss_list_io.cxx ss_list_io.h
  synthetic_mha.cxx synthetic_mha.h
  synthetic_vf.cxx synthetic_vf.h
  thumbnail.cxx thumbnail.h
  xio_ct.cxx xio_ct.h
  xio_demographic.cxx xio_demographic.h
  xio_dir.cxx xio_dir.h
  xio_dose.cxx xio_dose.h
  xio_io.cxx xio_io.h
  xio_plan.cxx xio_plan.h
  xio_structures.cxx xio_structures.h
  xform.cxx xform.h 
  xform_convert.cxx xform_convert.h 
  xform_legacy.cxx xform_legacy.h
  xform_point.cxx xform_point.h 
  )

if (ITK_FOUND AND GDCM_VERSION_1)
  set (PLASTIMATCH_LIBRARY_SRC
    ${PLASTIMATCH_LIBRARY_SRC}
    gdcm_dose.cxx gdcm_dose.h
    gdcm_file.cxx gdcm_file.h
    gdcm1_rdd.cxx gdcm1_rdd.h 
    gdcm_rtss.cxx gdcm_rtss.h
    gdcm_series.cxx gdcm_series.h
    gdcm_series_helper_2.cxx gdcm_series_helper_2.h
    gdcm1_util.cxx gdcm1_util.h
    )
endif ()

if (ITK_FOUND AND GDCM_VERSION_2)
  set (PLASTIMATCH_LIBRARY_SRC
    ${PLASTIMATCH_LIBRARY_SRC}
    gdcm2_util.cxx gdcm2_util.h
    )
endif ()

##-----------------------------------------------------------------------------
##  SOURCE FILES (EXECUTABLES)
##-----------------------------------------------------------------------------
SET (BRAGG_CURVE_SRC
  bragg_curve_main.c 
  bragg_curve_opts.c bragg_curve_opts.h
  )
SET (BSPLINE_SRC
  bspline_main.cxx 
  )
SET (CHECK_GRAD_SRC
  check_grad.cxx
  check_grad_opts.cxx check_grad_opts.h 
  )
SET (REG_SRC
  reg_main.cxx
  reg_opts.cxx reg_opts.h 
  )
SET (COMPUTE_DISTANCE_SRC
  compute_distance.cxx compute_distance.h
  closest_point.cxx 
  read_obj.cxx
  )
SET (COMPUTE_MEAN_IMAGE_SRC
  compute_mean_image_main.cxx
  )
SET (EXTRACT_SLICE_SRC
  extract_slice.cxx
  )
SET (CUDA_PROBE_SRC
  cuda_probe_main.c
  cuda_probe.cu
  cuda_probe.h
  )
IF (CUDA_FOUND)
  CUDA_COMPILE(CUDA_PROBE_WRAPPERS cuda_probe.cu)
  SET (CUDA_PROBE_SRC
    ${CUDA_PROBE_SRC}
    ${CUDA_PROBE_WRAPPERS}
    )
ENDIF ()
SET (DEMONS_SRC
  demons_main.c
  demons_opts.c demons_opts.h
  )
SET (DICE_STATS_SRC
  dice_stats_main.cxx
  dice_statistics.cxx dice_statistics.h
  hausdorff_statistics.cxx hausdorff_statistics.h
  )
SET (DICOM_INFO_SRC
    dicom_info.cxx)
SET (DICOM_SERIES_TEST_SRC
  dicom_series_test.cxx)
SET (DICOM_UID_SRC
  dicom_uid_main.cxx
  dicom_uid.cxx
  dicom_uid.h
  )
SET (DISTANCE_MAP_SRC
  distance_map.cxx
  )
SET (DLIB_TRAIN_SRC 
  dlib_train.cxx
  )
SET (DRR_SRC
  drr_main.c 
  drr_opts.c drr_opts.h
  )
SET (EXTRACT_CONTOUR_SRC
  extract_contour.cxx
  )
SET (FDK_SRC
  fdk_main.c 
  fdk_opts.c fdk_opts.h
  fdk_util.c fdk_util.h
)
IF (BROOK_FOUND)
  SET (FDK_SRC
    ${FDK_SRC}
    fdk_brook.cpp
    fdk_brook_kernel.br
    )
ENDIF ()
SET (FINDSCU_SRC
  findscu.cc
  )
SET (HND_TO_PFM_SRC
  hnd_to_pfm.c
  )
SET (MEX_DRR_SRC
  mex_drr.c
  )
SET (MHA_TO_RTOG_DOSE_SRC
  mha_to_rtog_dose.cxx
  )
SET (MONDOSHOT_SRC
  dicom_uid.cxx dicom_uid.h
  mondoshot_dicom.cpp mondoshot_dicom.h
  mondoshot_main.cpp mondoshot_main.h
  )
SET (MONDOSHOT_RELAY_SRC
  mondoshot_relay_main.cpp
  )
SET (MOVESCU_SRC
  movescu.cc
  )
SET (OPENCL_PROBE_SRC
  opencl_probe_main.c
  )
IF (OPENCL_FOUND)
  PLM_ADD_OPENCL_FILE (GPUIT_LIBRARY_SRC opencl_probe.cl)
ENDIF ()
SET (PLASTIMATCH_SRC
  pcmd_add.cxx pcmd_add.h
  pcmd_adjust.cxx pcmd_adjust.h
  pcmd_autolabel_train.cxx pcmd_autolabel_train.h
  pcmd_autolabel.cxx pcmd_autolabel.h
  pcmd_compare.cxx pcmd_compare.h
  pcmd_compose.cxx pcmd_compose.h
  pcmd_crop.cxx pcmd_crop.h
  pcmd_diff.cxx pcmd_diff.h
  pcmd_dvh.cxx pcmd_dvh.h
  pcmd_xio_dvh.cxx pcmd_xio_dvh.h
  pcmd_header.cxx pcmd_header.h
  pcmd_mask.cxx pcmd_mask.h
  pcmd_resample.cxx pcmd_resample.h
  pcmd_segment.cxx pcmd_segment.h
  pcmd_stats.cxx pcmd_stats.h
  pcmd_thumbnail.cxx pcmd_thumbnail.h 
  pcmd_warp_dij.cxx 
  pcmd_warp.cxx pcmd_warp.h 
  pcmd_warp_pointset.cxx 
  pcmd_xf_convert.cxx pcmd_xf_convert.h 
  plastimatch_main.cxx
  )
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
  plastimatch-slicer-landwarp.cxx
  )
SET (PLASTIMATCH_SLICER_SYNTH_SRC
  plastimatch_slicer_synth.cxx
  )
SET (PLASTIMATCH_QT_SRC
  # pqt_application.cxx pqt_application.h
  pqt_data_source_dialog.cxx pqt_data_source_dialog.h
  pqt_data_source_list_model.cxx pqt_data_source_list_model.h
  pqt_database.cxx pqt_database.h
  pqt_findscu.cxx pqt_findscu.h
  pqt_main.cxx
  pqt_main_window.cxx pqt_main_window.h
  pqt_patient_list_model.cxx pqt_patient_list_model.h
  )
IF (QT4_FOUND AND QT_QTSQL_FOUND)
  QT4_WRAP_CPP (PLASTIMATCH_QT_SRC 
    #    pqt_application.h
    pqt_data_source_dialog.h
    pqt_data_source_list_model.h
    pqt_main_window.h 
    pqt_patient_list_model.h
    )
  QT4_WRAP_UI (PLASTIMATCH_QT_SRC 
    pqt_data_source_dialog.ui
    pqt_main_window.ui
    )
ENDIF ()
SET (PROTON_DOSE_SRC
  proton_dose_main.c 
  proton_dose_opts.c proton_dose_opts.h
  )
SET (LANDMARK_WARP_SRC
  landmark_warp_main.cxx
  )
SET (RANSAC_TEST_SRC 
  ransac_test.cxx
  )
SET (RTOG_TO_MHA_SRC
  exchkeys.cxx exchkeys.h
  rtog_to_mha.cxx
  )
SET (SHUFFLE_MHA_SRC
  shuffle_mha_main.cxx
  )
SET (SIFTnD_SRC
  siftnD.cxx
  )
SET (SYNTHETIC_MHA_SRC
  synthetic_mha_main.cxx 
  )
SET (SYNTHETIC_VF_SRC
  synthetic_vf_main.cxx 
  )
SET (TEST2_SRC
  test2.cxx
  )
SET (VMATRIX_SRC
  Vmatrix.c
  )
SET (XYZINT_SRC
  xyzInt.c
  )



SET (VF_INVERT_SRC
  vf_invert.cxx
  )
SET (XF_TO_EMPIREFMT_SRC
  xf_to_empirefmt.cxx
  )

##-----------------------------------------------------------------------------
##  SOURCE FILE PROPERTIES
##-----------------------------------------------------------------------------
IF (OPENMP_FOUND)
  SET_SOURCE_FILES_PROPERTIES (bspline.cxx
    PROPERTIES COMPILE_FLAGS ${OPENMP_FLAGS})
  SET_SOURCE_FILES_PROPERTIES (bspline_mi.cxx
    PROPERTIES COMPILE_FLAGS ${OPENMP_FLAGS})
  SET_SOURCE_FILES_PROPERTIES (bspline_mse.cxx
    PROPERTIES COMPILE_FLAGS ${OPENMP_FLAGS})
  SET_SOURCE_FILES_PROPERTIES (drr.cxx
    PROPERTIES COMPILE_FLAGS ${OPENMP_FLAGS})
  SET_SOURCE_FILES_PROPERTIES (fdk.cxx
    PROPERTIES COMPILE_FLAGS ${OPENMP_FLAGS})
  SET_SOURCE_FILES_PROPERTIES (openmp_test.c
    PROPERTIES COMPILE_FLAGS ${OPENMP_FLAGS})
ENDIF ()

if (SSE2_FOUND AND NOT BUILD_AGAINST_SLICER3)
  PLM_SET_SSE2_FLAGS (bspline.cxx bspline_mi.cxx bspline_mse.cxx)
endif ()

##-----------------------------------------------------------------------------
##  Do we build a full plastimatch?
##-----------------------------------------------------------------------------
if (ITK_FOUND AND (NOT ITK_USE_SYSTEM_GDCM OR DCMTK_VERSION_36))
  set (FULL_PLASTIMATCH_BUILD 1)
else ()
  message (WARNING "Your plastimatch build is incomplete.  Please build ITK with GDCM 1.X.")
  set (FULL_PLASTIMATCH_BUILD 0)
endif ()

##-----------------------------------------------------------------------------
##  Library targets
##-----------------------------------------------------------------------------
PLM_ADD_LIBRARY (
  plmsys
  "${PLMSYS_LIBRARY_SRC}" 
  "${PLMSYS_LIBRARY_DEPENDENCIES}"
  "")

IF (F2C_LIBRARY AND UNIX)
  PLM_ADD_LIBRARY (f2c_helper "${F2C_HELPER_LIBRARY_SRC}" "" "")
ENDIF ()

PLM_ADD_LIBRARY (
  gpuit 
  "${GPUIT_LIBRARY_SRC}" 
  "${GPUIT_LIBRARY_DEPENDENCIES}"
  "${GPUIT_LIB_LDFLAGS}")

# JAS 2010.11.23
# Because the plugins are loaded at runtime, CMake doesn't know
# that they are needed by everything linked against gpuit.  So, we tell it.
IF (BUILD_SHARED_LIBS)      # << Dynamic loading
  IF (CUDA_FOUND)
    ADD_DEPENDENCIES (gpuit plmcuda)
  ENDIF ()
  IF (OPENCL_FOUND)
    ADD_DEPENDENCIES (gpuit plmopencl)
  ENDIF ()
ENDIF ()

if (FULL_PLASTIMATCH_BUILD)
  # Despite claims to the contrary, you are not allowed to have a library
  # with the same name as the executable.  What happens is the executable 
  # tries to create a temporary library with the same name.  [MSVC 2005]
  # That is why the library is called plastimatch1 instead of plastimatch.
  set (PLASTIMATCH_LIBRARY_DEPENDENCIES ${ITK_LIBRARIES} ${GPUIT_LIBRARIES})

  # Ugh.  ITK is stupid.  They quietly dropped support for cygwin. 
  # However, you can still get 3.20.0 to work.  You need to run 
  # "make ITKFEM" after running "make" to workaround ITK's broken CMake 
  # script:
  #  http://old.nabble.com/compile-error-using-cygwin-td30879187.html
  # See also:
  #  http://public.kitware.com/Bug/view.php?id=11659
  if (CYGWIN)
    set (PLASTIMATCH_LIBRARY_DEPENDENCIES 
      ${PLASTIMATCH_LIBRARY_DEPENDENCIES} gdi32)
  endif ()

  plm_add_library (
    plastimatch1 
    "${PLASTIMATCH_LIBRARY_SRC}"
    "${PLASTIMATCH_LIBRARY_DEPENDENCIES}"
    "")
endif ()

if (WIN32 AND SLICER_FOUND)
  # The slicer plugin on windows can't find its dll easily.  Copy these 
  # over to the lib directory.
  SET (INDIR
    "${CMAKE_BINARY_DIR}/${CMAKE_CFG_INTDIR}")
  SET (OUTDIR
    "${CMAKE_BINARY_DIR}/${Slicer3_INSTALL_PLUGINS_BIN_DIR}/${CMAKE_CFG_INTDIR}")

  PLM_SLICER_COPY_DLL (plastimatch_slicer_copy_plastimatch1
    "${INDIR}/plastimatch1.dll"
    "${OUTDIR}/plastimatch1.dll"
    plastimatch1
    )
  PLM_SLICER_COPY_DLL (plastimatch_slicer_copy_gpuit
    "${INDIR}/gpuit.dll"
    "${OUTDIR}/gpuit.dll"
    gpuit
    )
  PLM_SLICER_COPY_DLL (plastimatch_slicer_copy_plmsys
    "${INDIR}/plmsys.dll"
    "${OUTDIR}/plmsys.dll"
    plmsys
    )
  if (CUDA_FOUND)
    PLM_SLICER_COPY_DLL (plastimatch_slicer_copy_plmcuda
      "${INDIR}/plmcuda.dll"
      "${OUTDIR}/plmcuda.dll"
      plmcuda
      )
  endif ()
  if (OPENCL_FOUND)
    PLM_SLICER_COPY_DLL (plastimatch_slicer_copy_plmopencl
      "${INDIR}/plmopencl.dll"
      "${OUTDIR}/plmopencl.dll"
      plmopencl
      )
  endif ()
endif ()

##-----------------------------------------------------------------------------
##  Executable targets
##-----------------------------------------------------------------------------
PLM_ADD_EXECUTABLE (bragg_curve "${BRAGG_CURVE_SRC}" "${GPUIT_LIBRARIES}" 
  "${GPUIT_EXE_LDFLAGS}" 0)
PLM_ADD_EXECUTABLE (bspline "${BSPLINE_SRC}" "${GPUIT_LIBRARIES}" 
  "${PLMCUDA_EXE_LDFLAGS}" 0)
PLM_ADD_EXECUTABLE (check_grad "${CHECK_GRAD_SRC}" "${GPUIT_LIBRARIES}" 
  "${GPUIT_EXE_LDFLAGS}" 0)
PLM_ADD_EXECUTABLE (demons "${DEMONS_SRC}" "${GPUIT_LIBRARIES}" 
  "${PLMCUDA_EXE_LDFLAGS}" 0)
PLM_ADD_EXECUTABLE (drr "${DRR_SRC}" "${GPUIT_LIBRARIES}" 
  "${PLMCUDA_EXE_LDFLAGS}" 1)
PLM_ADD_EXECUTABLE (dlib_train "${DLIB_TRAIN_SRC}" "${GPUIT_LIBRARIES}" 
  "${PLMCUDA_EXE_LDFLAGS}" 1)
PLM_ADD_EXECUTABLE (fdk "${FDK_SRC}" "${GPUIT_LIBRARIES}" 
  "${PLMCUDA_EXE_LDFLAGS}" 1)
PLM_ADD_EXECUTABLE (reg "${REG_SRC}" "${GPUIT_LIBRARIES}" 
  "${GPUIT_EXE_LDFLAGS}" 0)
PLM_ADD_EXECUTABLE (hnd_to_pfm "${HND_TO_PFM_SRC}" "${GPUIT_LIBRARIES}" 
  "${GPUIT_EXE_LDFLAGS}" 1)
PLM_ADD_EXECUTABLE (mha_to_raw mha_to_raw.c "" "" 1)
PLM_ADD_EXECUTABLE (proton_dose "${PROTON_DOSE_SRC}" "${GPUIT_LIBRARIES}" 
  "${GPUIT_EXE_LDFLAGS}" 0)
PLM_ADD_EXECUTABLE (raw_to_mha raw_to_mha.c "" "" 1)
PLM_ADD_EXECUTABLE (test2 "${TEST2_SRC}" 
  "" "" 0)
PLM_ADD_EXECUTABLE (Vmatrix "${VMATRIX_SRC}" 
  "" "" 0)
PLM_ADD_EXECUTABLE (xyzInt "${XYZINT_SRC}" 
  "" "" 1)



if (FULL_PLASTIMATCH_BUILD)
  MESSAGE (STATUS "PLASTIMATCH_LIBS: ${PLASTIMATCH_LIBS}")
  PLM_ADD_EXECUTABLE (compute_distance "${COMPUTE_DISTANCE_SRC}" 
    "${PLASTIMATCH_LIBS}" "${GPUIT_EXE_LDFLAGS}" 1)
  PLM_ADD_EXECUTABLE (compute_mean_image "${COMPUTE_MEAN_IMAGE_SRC}"
    "${PLASTIMATCH_LIBS}" "${GPUIT_EXE_LDFLAGS}" 0)
  PLM_ADD_EXECUTABLE (extract_slice "${EXTRACT_SLICE_SRC}" 
    "${PLASTIMATCH_LIBS}" "${GPUIT_EXE_LDFLAGS}" 0)
  PLM_ADD_EXECUTABLE (dice_stats "${DICE_STATS_SRC}"
    "${PLASTIMATCH_LIBS}" "${GPUIT_EXE_LDFLAGS}" 1)
  PLM_ADD_EXECUTABLE (dicom_info "${DICOM_INFO_SRC}"
    "${PLASTIMATCH_LIBS}" "${GPUIT_EXE_LDFLAGS}" 
    ${INSTALL_IF_NOT_DEBIAN})
  PLM_ADD_EXECUTABLE (dicom_series_test "${DICOM_SERIES_TEST_SRC}"
    "${PLASTIMATCH_LIBS}" "${GPUIT_EXE_LDFLAGS}" 0)
  PLM_ADD_EXECUTABLE (distance_map "${DISTANCE_MAP_SRC}"
    "${PLASTIMATCH_LIBS}" "${GPUIT_EXE_LDFLAGS}" 0)
  PLM_ADD_EXECUTABLE (extract_contour "${EXTRACT_CONTOUR_SRC}"
    "${PLASTIMATCH_LIBS}" "${GPUIT_EXE_LDFLAGS}" 0)
  PLM_ADD_EXECUTABLE (landmark_warp "${LANDMARK_WARP_SRC}" 
    "${PLASTIMATCH_LIBS}" "${GPUIT_EXE_LDFLAGS}" 1)
  PLM_ADD_EXECUTABLE (merge_vfs merge_vector_fields.cxx
    "${PLASTIMATCH_LIBS}" "${GPUIT_EXE_LDFLAGS}" 1)
  PLM_ADD_EXECUTABLE (mha_to_rtog_dose "${MHA_TO_RTOG_DOSE_SRC}"
    "${PLASTIMATCH_LIBS}" "${GPUIT_EXE_LDFLAGS}" 0)
  PLM_ADD_EXECUTABLE (plastimatch "${PLASTIMATCH_SRC}"
    "${PLASTIMATCH_LIBS}" "${PLMCUDA_EXE_LDFLAGS}" 1)
  IF (QT4_FOUND AND QT_QTSQL_FOUND)
    SET (PLASTIMATCH_QT_LIBRARIES ${PLASTIMATCH_LIBS} 
      ${QT_LIBRARIES} ${QT_QTSQL_LIBRARIES})
    PLM_ADD_EXECUTABLE (plastimatch_qt "${PLASTIMATCH_QT_SRC}"
      "${PLASTIMATCH_QT_LIBRARIES}" "${GPUIT_EXE_LDFLAGS}" 0)
  ENDIF ()
  PLM_ADD_EXECUTABLE (ransac_test "${RANSAC_TEST_SRC}" 
    "${PLASTIMATCH_LIBS}" "${GPUIT_EXE_LDFLAGS}" 0)
  PLM_ADD_EXECUTABLE (rtog_to_mha "${RTOG_TO_MHA_SRC}"
    "${PLASTIMATCH_LIBS}" "${GPUIT_EXE_LDFLAGS}" 0)
  PLM_ADD_EXECUTABLE (shuffle_mha "${SHUFFLE_MHA_SRC}"
    "${PLASTIMATCH_LIBS}" "${GPUIT_EXE_LDFLAGS}" 1)
  PLM_ADD_EXECUTABLE (siftnD "${SIFTnD_SRC}"
    "${PLASTIMATCH_LIBS}" "${GPUIT_EXE_LDFLAGS}" 1)
  PLM_ADD_EXECUTABLE (synthetic_mha "${SYNTHETIC_MHA_SRC}"
    "${PLASTIMATCH_LIBS}" "${GPUIT_EXE_LDFLAGS}" 1)
  PLM_ADD_EXECUTABLE (synthetic_vf "${SYNTHETIC_VF_SRC}"
    "${PLASTIMATCH_LIBS}" "${GPUIT_EXE_LDFLAGS}" 1)
  PLM_ADD_EXECUTABLE (union_mask union_mask.cxx
    "${PLASTIMATCH_LIBS}" "${GPUIT_EXE_LDFLAGS}" 1)
  PLM_ADD_EXECUTABLE (vf_invert "${VF_INVERT_SRC}"
    "${PLASTIMATCH_LIBS}" "${GPUIT_EXE_LDFLAGS}" 1)
  PLM_ADD_EXECUTABLE (xf_to_EMPIREFMT "${XF_TO_EMPIREFMT_SRC}"
    "${PLASTIMATCH_LIBS}" "${GPUIT_EXE_LDFLAGS}" 0)
ENDIF ()

# executables that require cuda
IF (CUDA_FOUND)
  PLM_ADD_EXECUTABLE (cuda_probe 
    "${CUDA_PROBE_SRC}" "${CUDA_LIBRARIES}" "" 1)
ENDIF ()

# executables that require opencl
IF (OPENCL_FOUND)
  PLM_ADD_EXECUTABLE (opencl_probe 
    "${OPENCL_PROBE_SRC}"  "${GPUIT_LIBRARIES}" "${GPUIT_EXE_LDFLAGS}" 1)
ENDIF ()

# dicom_uid requires dcmtk
IF (DCMTK_FOUND)
  PLM_ADD_EXECUTABLE (dicom_uid 
    "${DICOM_UID_SRC}" "${DCMTK_LIBRARIES}" "" 
    ${INSTALL_IF_NOT_DEBIAN})
  IF (NOT DCMTK_VERSION_NUMBER GREATER 354)
    PLM_ADD_EXECUTABLE (findscu "${FINDSCU_SRC}" "${DCMTK_LIBRARIES}" "" 0)
    PLM_ADD_EXECUTABLE (movescu "${MOVESCU_SRC}" "${DCMTK_LIBRARIES}" "" 0)
  ENDIF ()
ENDIF ()

# mondoshot requires WIN32, wx, dcmtk, sqlite3
IF (WIN32 AND NOT CYGWIN AND wxWidgets_FOUND AND DCMTK_FOUND)
  ADD_EXECUTABLE (mondoshot WIN32 ${MONDOSHOT_SRC})
  TARGET_LINK_LIBRARIES (mondoshot 
    ${wxWidgets_LIBRARIES} ${DCMTK_LIBRARIES} ${SQLITE_LIBRARIES})
  IF (NOT BUILD_AGAINST_SLICER3)
    INSTALL (TARGETS mondoshot DESTINATION bin)
  ENDIF ()
ENDIF ()

# matlab mex files
IF (MATLAB_FOUND AND PLM_BUILD_MEX_TEST)
  MEX_TARGET (mex_drr 
    "${MEX_DRR_SRC}" "${GPUIT_LIBRARIES}" "${GPUIT_EXE_LDFLAGS}")
ENDIF ()

##-----------------------------------------------------------------------------
##  Slicer loadable module (in development)
##-----------------------------------------------------------------------------
IF (SLICER_FOUND AND NOT SLICER_IS_SLICER3 AND PLM_BUILD_SLICER_LOADABLE)

  SET (CMAKE_MODULE_PATH "${Slicer_DIR}/CMake" ${CMAKE_MODULE_PATH})
  #INCLUDE ("${Slicer_CMAKE_DIR}/SlicerQTModuleMacros.cmake")
  #INCLUDE ("${Slicer_CMAKE_DIR}/SlicerParseArgumentsMacro.cmake")
  ADD_SUBDIRECTORY (slicer_loadable)

  IF (COMMENTOUT)
    CONFIGURE_FILE (
      ${CMAKE_CURRENT_SOURCE_DIR}/vtkPlastimatchLoadableConfigure.h.in 
      ${CMAKE_CURRENT_BINARY_DIR}/vtkPlastimatchLoadableConfigure.h
      )
    set (PlastimatchLoadable_SRCS
      vtkPlastimatchLoadableLogic.cxx 
      vtkPlastimatchLoadableGUI.cxx 
      vtkMRMLPlastimatchLoadableNode.cxx 
      )

    include("${VTK_CMAKE_DIR}/vtkWrapTcl.cmake")
    vtk_wrap_tcl3 (PlastimatchLoadable
      PlastimatchLoadable_TCL_SRCS 
      "${PlastimatchLoadable_SRCS}" "")
    MESSAGE (STATUS "TCL SOURCES: ${PlastimatchLoadable_TCL_SRCS}")

    GENERATELM (PlastimatchLoadable_SRCS PlastimatchLoadable.xml)

    PLM_ADD_SLICER_MODULE (PlastimatchLoadable
      "${PlastimatchLoadable_SRCS};${PlastimatchLoadable_TCL_SRCS}" 
      "${PLASTIMATCH_LIBS}")

    ADD_LIBRARY (PlastimatchLoadable
      ${PlastimatchLoadable_SRCS}
      ${PlastimatchLoadable_TCL_SRCS})
    SLICER3_SET_MODULES_OUTPUT_PATH (PlastimatchLoadable)
    TARGET_LINK_LIBRARIES (PlastimatchLoadable
	${Slicer_Libs_LIBRARIES}
	${Slicer_Base_LIBRARIES}
	${KWWidgets_LIBRARIES}
	${ITK_LIBRARIES})
      #      ${PLASTIMATCH_LIBS})
    SLICER3_INSTALL_MODULES (PlastimatchLoadable)
  ENDIF (COMMENTOUT)
ENDIF ()

##-----------------------------------------------------------------------------
##  Simple test files
##-----------------------------------------------------------------------------
# Test executable -- cuda
IF (CUDA_FOUND AND PLM_BUILD_CUDA_TEST)
  CUDA_COMPILE (CUDA_TEST_WRAPPERS cuda_test.cu)
  SET (CUDA_TEST_SRC ${CUDA_TEST_WRAPPERS} cuda_test.cu)
  MESSAGE (STATUS "CUDA_TEST_SRC: ${CUDA_TEST_SRC}")
  PLM_ADD_EXECUTABLE (cuda_test 
    "${CUDA_TEST_SRC}" "${CUDA_LIBRARIES}" "" 0)
  SET_TARGET_PROPERTIES (cuda_test PROPERTIES LINKER_LANGUAGE CXX)
ENDIF ()

# Test executable -- dlib
IF (PLM_BUILD_DLIB_TEST)
  SET (DLIB_TEST_SRC dlib_test.cxx)
  SET (DLIB_TEST_LIBS "")
  PLM_ADD_EXECUTABLE (dlib_test 
    "${DLIB_TEST_SRC}" "${DLIB_TEST_LIBS}" "" 0)
ENDIF ()

# Test executable -- fann
IF (FANN_FOUND AND PLM_BUILD_FANN_TEST)
  SET (FANN_TEST_SRC fann_test.c)
  SET (FANN_TEST_LIBS 
    ${GPUIT_LIBRARIES} ${FANN_LIBRARIES})
  PLM_ADD_EXECUTABLE (fann_test 
    "${FANN_TEST_SRC}" "${FANN_TEST_LIBS}" "${OPENMP_LDFLAGS}" 0)
ENDIF ()

# Test executable -- mex
IF (PLM_BUILD_MEX_TEST)
  IF (OCTAVE_FOUND)
    ADD_CUSTOM_COMMAND (
      OUTPUT "${CMAKE_BINARY_DIR}/mex_test.mex"
      COMMAND ${OCTAVE_MKOCTFILE} --mex 
      -o "${CMAKE_BINARY_DIR}/mex_test.mex" 
      "${CMAKE_SOURCE_DIR}/mex_test.c"
      DEPENDS "${CMAKE_SOURCE_DIR}/mex_test.c")
    ADD_CUSTOM_TARGET (oct_mex_test
      DEPENDS "${CMAKE_BINARY_DIR}/mex_test.mex")
  ENDIF ()
  IF (MATLAB_FOUND)
    FILE (WRITE "${CMAKE_BINARY_DIR}/compile_mex_test.m"
      "mex \"${CMAKE_SOURCE_DIR}/mex_test.c\";exit;\n")
    FILE (WRITE "${CMAKE_BINARY_DIR}/mex_test.cmake"
      "EXECUTE_PROCESS (COMMAND ${MATLAB_EXE} -nosplash -nodesktop -nojvm
       -r compile_mex_test
       RESULT_VARIABLE RESULT
       OUTPUT_VARIABLE STDOUT
       ERROR_VARIABLE STDERR)\n")
    ADD_CUSTOM_COMMAND (
      OUTPUT "${CMAKE_BINARY_DIR}/mex_test${MATLAB_MEXEXT}"
      COMMAND ${CMAKE_COMMAND} -P "${CMAKE_BINARY_DIR}/mex_test.cmake"
      DEPENDS "${CMAKE_SOURCE_DIR}/mex_test.c")
    ADD_CUSTOM_TARGET (mat_mex_test
      DEPENDS "${CMAKE_BINARY_DIR}/mex_test${MATLAB_MEXEXT}")
    TARGET_LINK_LIBRARIES (mex_test ${MATLAB_LIBRARIES})
  ENDIF ()
ENDIF ()

# Test executable -- nlopt
IF (PLM_BUILD_NLOPT_TEST AND NLOPT_FOUND)
  SET (NLOPT_TEST_SRC nlopt_test.c)
  SET (NLOPT_TEST_LIBS 
    ${GPUIT_LIBRARIES} ${NLOPT_LIBRARIES})
  PLM_ADD_EXECUTABLE (nlopt_test "${NLOPT_TEST_SRC}" "${NLOPT_TEST_LIBS}" 
    "${OPENMP_LDFLAGS}" 0)
  SET_TARGET_PROPERTIES (nlopt_test PROPERTIES LINKER_LANGUAGE CXX)
ENDIF ()

# Test executable -- opencl
IF (PLM_BUILD_OPENCL_TEST AND OPENCL_FOUND)
  SET (OPENCL_TEST_SRC opencl_test.cxx opencl_test.h opencl_test.cl
    "${CMAKE_BINARY_DIR}/opencl_test.cl"
    )
  SET (OPENCL_TEST_LIBS ${OPENCL_LIBRARIES} ${PLASTIMATCH_LIBS})
  #SET (OPENCL_TEST_LIBS ${OPENCL_LIBRARIES} ${GPUIT_LIBRARIES})
  PLM_ADD_EXECUTABLE (opencl_test "${OPENCL_TEST_SRC}" 
    "${OPENCL_TEST_LIBS}" "" 0)
  #SET_TARGET_PROPERTIES (opencl_test PROPERTIES LINKER_LANGUAGE CXX)

  # I don't yet know how to bundle the .cl file within the executable.
  # Therefore, copy the .cl into binary directory.
  ADD_CUSTOM_COMMAND (
    OUTPUT "${CMAKE_BINARY_DIR}/opencl_test.cl" 
    COMMAND ${CMAKE_COMMAND} "-E" "copy" 
    "${CMAKE_CURRENT_SOURCE_DIR}/opencl_test.cl" 
    "${CMAKE_BINARY_DIR}/opencl_test.cl" 
    DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/opencl_test.cl")
ENDIF ()

# Test executable -- openmp
IF (PLM_BUILD_OPENMP_TEST AND OPENMP_FOUND)
  SET (OPENMP_TEST_SRC openmp_test.c)
  PLM_ADD_EXECUTABLE (openmp_test "${OPENMP_TEST_SRC}" "${GPUIT_LIBRARIES}" 
    "${OPENMP_LDFLAGS}" 0)
ENDIF ()

# Test executable -- cuda_tex_test
if (PLM_BUILD_CUDA_TEST AND CUDA_FOUND)
  add_subdirectory (CUDA_tex_test)
endif ()

# Test executable -- qt
IF (PLM_BUILD_QT_TEST AND QT4_FOUND)
  SET (QT_TEST_SRC qt_test.cxx)
  PLM_ADD_EXECUTABLE (qt_test "${QT_TEST_SRC}" "${QT_LIBRARIES}" "" 0)
ENDIF ()

## -------------------------------------------------------------------------
## Unit test targets
## -------------------------------------------------------------------------
PLM_ADD_EXECUTABLE (bspline_test_1 bspline_test_1.cxx
  "${GPUIT_LIBRARIES}" "${GPUIT_EXE_LDFLAGS}" 0)
IF (ITK_FOUND)
  PLM_ADD_EXECUTABLE (plm_drr_api_a plm_drr_api_a.cxx
    "${PLASTIMATCH_LIBS}" "${GPUIT_EXE_LDFLAGS}" 0)
  PLM_ADD_EXECUTABLE (plm_registration_api_a plm_registration_api_a.cxx
    "${PLASTIMATCH_LIBS}" "${GPUIT_EXE_LDFLAGS}" 0)
ENDIF ()
