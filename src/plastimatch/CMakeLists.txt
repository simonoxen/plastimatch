##-----------------------------------------------------------------------------
##  Welcome to the Plastimatch CMakeLists.txt file
##-----------------------------------------------------------------------------
##  See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
##-----------------------------------------------------------------------------
project (src_plastimatch)

##-----------------------------------------------------------------------------
##  Local options
##-----------------------------------------------------------------------------
# Building little test programs
option (PLM_BUILD_CUDA_TEST "Build CUDA test program" OFF)
option (PLM_BUILD_DCMTK_TEST "Build DCMTK test program" OFF)
option (PLM_BUILD_DLIB_TEST "Build dlib test program" OFF)
option (PLM_BUILD_FANN_TEST "Build fann test program" OFF)
option (PLM_BUILD_GDCM1_TEST "Build GDCM test program" OFF)
option (PLM_BUILD_MEX_TEST "Build mex test program" OFF)
option (PLM_BUILD_NLOPT_TEST "Build NLopt test program" OFF)
option (PLM_BUILD_OPENCL_TEST "Build OpenCL test program" OFF)
option (PLM_BUILD_OPENMP_TEST "Build OpenMP test program" OFF)
option (PLM_BUILD_QT_TEST "Build Qt test program" OFF)

option (PLM_BUILD_SIFTND "Build siftnD program" ON)

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
include_directories (BEFORE ${CMAKE_CURRENT_SOURCE_DIR})
include_directories (BEFORE ${CMAKE_BINARY_DIR})
include_directories (BEFORE ${CMAKE_CURRENT_BINARY_DIR})
include_directories (AFTER ${BSTRLIB_INCLUDE_DIR})
include_directories (AFTER ${LIBLBFGS_INCLUDE_DIR})
include_directories (AFTER ${MSINTTYPES_INCLUDE_DIR})
if (SQLITE_FOUND)
  include_directories (AFTER ${SQLITE_INCLUDE_DIR})
endif ()
include_directories (AFTER ${CMAKE_SOURCE_DIR}/libs/itk-3.20.0)
include_directories (AFTER ${CMAKE_SOURCE_DIR}/libs/nSIFT)

# Don't do the below.  A user-built f2c directory contains ctype.h, which 
# conflicts with the C/C++ header of the same name.
#IF (PLM_USE_F2C)
#  INCLUDE_DIRECTORIES (BEFORE ${F2C_INCLUDE_DIR})
#ENDIF ()

if (CUDA_FOUND)
  include_directories (AFTER ${CUDA_INCLUDE_DIRS})
  #INCLUDE_DIRECTORIES (AFTER ${CUDA_CUT_INCLUDE_DIR})
endif ()

if (DCMTK_FOUND)
  include_directories (AFTER ${DCMTK_INCLUDE_DIR})
endif ()

include_directories (AFTER ${DLIB_INCLUDE_DIR})

if (FFTW_FOUND)
  include_directories (BEFORE ${FFTW_INCLUDE_DIR})
  link_directories (${FFTW_DIR})
endif ()

if (NOT HAVE_GETOPT_LONG)
  include_directories (AFTER "${CMAKE_SOURCE_DIR}/libs/getopt")
endif ()

if (OPENCL_FOUND)
  include_directories (BEFORE ${OPENCL_INCLUDE_DIR})
endif ()

if (MATLAB_FOUND)
  include_directories (AFTER ${MATLAB_INCLUDE_DIRS})
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

include_directories (AFTER ${RANSAC_INCLUDE_DIRS})

if (wxWidgets_FOUND)
  include_directories (BEFORE ${wxWidgets_INCLUDE_DIRS})
endif ()

##-----------------------------------------------------------------------------
##  PLMSYS/GPUIT/PLASTIMATCH LIBRARY SOURCE FILES
##-----------------------------------------------------------------------------
set (PLMSYS_LIBRARY_SRC
  file_util.cxx file_util.h
  logfile.c logfile.h
  interpolate.cxx interpolate.h
  plm_timer.cxx plm_timer.h
  print_and_exit.c print_and_exit.h 
  )

set (GPUIT_LIBRARY_SRC
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
  bspline_regularize_analytic.cxx bspline_regularize_analytic.h
  bspline_regularize_numeric.cxx bspline_regularize_numeric.h
  bspline_warp.cxx bspline_warp.h
  bspline_xform.cxx bspline_xform.h
  delayload.c delayload.h
  demons.cxx demons.h
  demons_cpu.cxx
  demons_misc.cxx demons_misc.h
  demons_state.cxx demons_state.h
  dir_list.cxx dir_list.h
  drr.cxx drr.h
  drr_trilin.cxx drr_trilin.h
  fdk.cxx fdk.h
  hnd_io.cxx hnd_io.h
  math_util.h
  mha_io.cxx mha_io.h
  rpl_volume.cxx rpl_volume.h
  plm_endian.cxx plm_endian.h 
  plm_fwrite.cxx plm_fwrite.h
  pointset.cxx pointset.h 
  proj_image.cxx proj_image.h
  proj_image_dir.cxx proj_image_dir.h
  proj_matrix.cxx proj_matrix.h 
  proton_dose.cxx proton_dose.h 
  pstring.h
  raw_pointset.cxx raw_pointset.h 
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
  volume_resample.cxx volume_resample.h
  xpm.cxx xpm.h
  )

if (CUDA_FOUND)
  set (PLMCUDA_LIBRARY_SRC
    bspline_cuda.cpp bspline_cuda.cu bspline_cuda.h
    cuda_kernel_util.cu cuda_kernel_util.h
    cuda_mem.cu cuda_mem.h
    cuda_util.cu cuda_util.h
    demons_cuda.cu demons_cuda.h
    demons_misc.cxx demons_misc.h
    drr_cuda.cpp drr_cuda.cu drr_cuda.h drr_cuda_p.h 
    fdk_cuda.cpp fdk_cuda.cu fdk_cuda_p.h fdk_cuda.h
    )

  if (BUILD_SHARED_LIBS)      # << Dynamic loading
    set (PLMCUDA_LIBRARY_SRC
      ${PLMCUDA_LIBRARY_SRC}
#      volume.cxx          # for demons_cuda.cu
#      volume.h            # for demons_cuda.cu
    )
    set (PLMCUDA_LIBRARY_DEPENDENCIES
      plmsys
    )
    cuda_add_library (plmcuda SHARED
      ${PLMCUDA_LIBRARY_SRC}
      )
    set_target_properties (plmcuda PROPERTIES 
      ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
      LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
      RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
    target_link_libraries (plmcuda
      ${PLMCUDA_LIBRARY_DEPENDENCIES}
      )
    install (TARGETS plmcuda
      RUNTIME DESTINATION bin
      LIBRARY DESTINATION lib
      )
  else ()        # << Static linking
    set (GPUIT_LIBRARY_SRC
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
    cuda_compile (CUDA_WRAPPERS ${PLM_CUDA_FILES})
    set (GPUIT_LIBRARY_SRC
      ${GPUIT_LIBRARY_SRC}
      ${CUDA_WRAPPERS}
      ${PLM_CUDA_FILES}
      )
  endif ()
endif ()

if (FFTW_FOUND)
  set (GPUIT_LIBRARY_SRC
    ${GPUIT_LIBRARY_SRC}
    ramp_filter.c ramp_filter.h
   )
endif ()

if (OPENCL_FOUND)
  # Define OpenCL Plugin (plmopencl)
  set (PLMOPENCL_LIBRARY_SRC
    autotune_opencl.cxx autotune_opencl.h
    opencl_probe.cxx opencl_probe.h opencl_probe.cl
    opencl_util.cxx opencl_util.h
    )

  if (BUILD_SHARED_LIBS)      # << Dynamic loading

    set (PLMOPENCL_LIBRARY_SRC
      ${PLMOPENCL_LIBRARY_SRC}
    )

    set (PLMOPENCL_LIBRARY_DEPENDENCIES
      plmsys
    )

    # OpenCL Plugin is ALWAYS shared!
    add_library (plmopencl SHARED
      ${PLMOPENCL_LIBRARY_SRC}
      )
    set_target_properties (plmopencl PROPERTIES 
      ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
      LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
      RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")

    # Be sure to link against libOpenCL and other depends
    target_link_libraries (plmopencl
      ${OPENCL_LIBRARIES}
      ${PLMOPENCL_LIBRARY_DEPENDENCIES}
      )

    install (TARGETS plmopencl
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION lib
    )

  else ()        # << Static linking

    set (GPUIT_LIBRARY_SRC
      ${GPUIT_LIBRARY_SRC}
      ${PLMOPENCL_LIBRARY_SRC}
    )

  endif ()

  set (GPUIT_LIBRARY_SRC
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
  plm_add_opencl_file (GPUIT_LIBRARY_SRC demons_opencl.cl)
  plm_add_opencl_file (GPUIT_LIBRARY_SRC drr_opencl.cl)
  plm_add_opencl_file (GPUIT_LIBRARY_SRC fdk_opencl.cl)

endif ()

if (F2C_LIBRARY)
  set (F2C_HELPER_LIBRARY_SRC
    main__.c
    )
endif ()

set (PLASTIMATCH1_LIBRARY_SRC
  autolabel_trainer.cxx autolabel_trainer.h 
  autolabel_ransac_est.cxx autolabel_ransac_est.h 
  astroid_dose.cxx astroid_dose.h
  cxt_extract.cxx cxt_extract.h
  cxt_io.cxx cxt_io.h 
  dcm_util.cxx dcm_util.h
  direction_cosines.h
  dlib_trainer.cxx dlib_trainer.h 
  dvh.cxx dvh.h
  gamma_analysis.cxx gamma_analysis.h
  gpuit_bspline.cxx gpuit_bspline.h 
  gpuit_demons.cxx gpuit_demons.h 
  img_metadata.cxx img_metadata.h 
  itkAndConstantToImageFilter.h
  itk_crop.cxx itk_crop.h 
  itk_demons.cxx itk_demons.h 
  itk_dicom_load.cxx itk_dicom_load.h 
  itk_dicom_save.cxx itk_dicom_save.h 
  itk_directions.cxx itk_directions.h 
  itk_gabor.cxx itk_gabor.h
  itk_image.cxx itk_image.h 
  itk_image_cast.cxx itk_image_cast.h 
  itk_image_load.h itk_image_load.txx
  itk_image_load_char.cxx
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
  itk_metadata.cxx itk_metadata.h 
  itk_pointset.cxx itk_pointset.h
  itk_optim.cxx itk_optim.h 
  itk_registration.cxx itk_registration.h 
  itk_resample.cxx itk_resample.h 
  itk_tps.cxx itk_tps.h 
  itk_volume_header.cxx itk_volume_header.h
  itk_warp.cxx itk_warp.h 
  landmark_warp.cxx landmark_warp.h 
  landmark_diff.cxx landmark_diff.h
  mc_dose.cxx mc_dose.h
  pcmd_diff.cxx pcmd_diff.h
  plm_clp.cxx plm_clp.h
  plm_drr_api.cxx plm_drr_api.h 
  plm_file_format.cxx plm_file_format.h 
  plm_image.cxx plm_image.h 
  plm_image_convert.cxx plm_image_convert.h 
  plm_image_header.cxx plm_image_header.h 
  plm_image_type.cxx plm_image_type.h 
  plm_int.h
  plm_parms.cxx plm_parms.h 
  plm_register_loadable.cxx plm_register_loadable.h 
  plm_registration_api.cxx plm_registration_api.h 
  plm_stages.cxx plm_stages.h 
  plm_version.h 
  plm_warp.cxx plm_warp.h 
  rasterizer.cxx rasterizer.h
  rasterize_slice.cxx rasterize_slice.h
  rbf_gauss.cxx rbf_gauss.h
  rbf_wendland.cxx rbf_wendland.h
  rbf_cluster.cxx rbf_cluster.h
  referenced_dicom_dir.cxx referenced_dicom_dir.h 
  registration_data.cxx registration_data.h
  rtds.cxx rtds.h
  rtds_dcmtk.cxx
  rtds_dicom.cxx rtds_dicom.h
  rtds_gdcm.cxx
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
  threshbox.cxx threshbox.h
  thumbnail.cxx thumbnail.h
  xio_ct.cxx xio_ct.h
  xio_demographic.cxx xio_demographic.h
  xio_dir.cxx xio_dir.h
  xio_dose.cxx xio_dose.h
  xio_studyset.cxx xio_studyset.h
  xio_patient.cxx xio_patient.h 
  xio_plan.cxx xio_plan.h
  xio_structures.cxx xio_structures.h
  xform.cxx xform.h 
  xform_convert.cxx xform_convert.h 
  xform_legacy.cxx xform_legacy.h
  xform_point.cxx xform_point.h 
  )

if (ITK_FOUND AND GDCM_VERSION_1)
  set (PLASTIMATCH1_LIBRARY_SRC
    ${PLASTIMATCH1_LIBRARY_SRC}
    gdcm1_dose.cxx gdcm1_dose.h
    gdcm1_file.cxx gdcm1_file.h
    gdcm1_rdd.cxx gdcm1_rdd.h 
    gdcm1_rtss.cxx gdcm1_rtss.h
    gdcm1_series.cxx gdcm1_series.h
    gdcm1_series_helper_2.cxx gdcm1_series_helper_2.h
    gdcm1_util.cxx gdcm1_util.h
    )
endif ()

if (ITK_FOUND AND GDCM_VERSION_2)
  set (PLASTIMATCH1_LIBRARY_SRC
    ${PLASTIMATCH1_LIBRARY_SRC}
    gdcm2_util.cxx gdcm2_util.h
    )
endif ()

if (DCMTK_FOUND AND NOT DCMTK_VERSION_STRING VERSION_LESS 3.6.0)
  set (PLASTIMATCH1_LIBRARY_SRC
    ${PLASTIMATCH1_LIBRARY_SRC}
    dcmtk_load.cxx dcmtk_load.h
    dcmtk_file.cxx dcmtk_file.h
    dcmtk_series.cxx dcmtk_series.h
    dcmtk_series_set.cxx dcmtk_series_set.h
    dcmtk_util.cxx dcmtk_util.h
    )
endif ()

##-----------------------------------------------------------------------------
##  SOURCE FILES (EXECUTABLES)
##-----------------------------------------------------------------------------
set (BRAGG_CURVE_SRC
  bragg_curve_main.c 
  bragg_curve_opts.c bragg_curve_opts.h
  )
set (BSPLINE_SRC
  bspline_main.cxx 
  )
set (CHECK_GRAD_SRC
  check_grad.cxx
  check_grad_opts.cxx check_grad_opts.h 
  )
set (REG_SRC
  reg_main.cxx
  reg_opts.cxx reg_opts.h 
  )
set (COMPUTE_DISTANCE_SRC
  compute_distance.cxx compute_distance.h
  closest_point.cxx 
  read_obj.cxx
  )
set (COMPUTE_MEAN_IMAGE_SRC
  compute_mean_image_main.cxx
  )
set (CUDA_PROBE_SRC
  cuda_probe_main.c
  cuda_probe.cu
  cuda_probe.h
  )
if (CUDA_FOUND)
  cuda_compile(CUDA_PROBE_WRAPPERS cuda_probe.cu)
  set (CUDA_PROBE_SRC
    ${CUDA_PROBE_SRC}
    ${CUDA_PROBE_WRAPPERS}
    )
endif ()
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
  dicom_info.cxx)
set (DICOM_UID_SRC
  dicom_uid_main.cxx
  dicom_uid.cxx
  dicom_uid.h
  )
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
set (EXTRACT_CONTOUR_SRC
  extract_contour.cxx
  )
set (FDK_SRC
  fdk_main.cxx 
  fdk_opts.cxx fdk_opts.h
  fdk_util.cxx fdk_util.h
)
set (HND_TO_PFM_SRC
  hnd_to_pfm.cxx 
  )
set (MEX_DRR_SRC
  mex_drr.c
  )
set (MHA_TO_RTOG_DOSE_SRC
  mha_to_rtog_dose.cxx
  )
set (MONDOSHOT_SRC
  dicom_uid.cxx dicom_uid.h
  mondoshot_dicom.cpp mondoshot_dicom.h
  mondoshot_main.cpp mondoshot_main.h
  )
set (OPENCL_PROBE_SRC
  opencl_probe_main.c
  )
if (OPENCL_FOUND)
  plm_add_opencl_file (GPUIT_LIBRARY_SRC opencl_probe.cl)
endif ()
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
  pcmd_segment.cxx pcmd_segment.h
  pcmd_stats.cxx pcmd_stats.h
  pcmd_synth.cxx pcmd_synth.h
  pcmd_thumbnail.cxx pcmd_thumbnail.h 
  pcmd_warp.cxx pcmd_warp.h 
  pcmd_warp_dij.cxx 
  pcmd_warp_pointset.cxx 
  pcmd_xio_dvh.cxx pcmd_xio_dvh.h
  pcmd_xf_convert.cxx pcmd_xf_convert.h 
  plastimatch_main.cxx
  )
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
set (PLASTIMATCH_QT_SRC
  # pqt_application.cxx pqt_application.h
  pqt_data_source_dialog.cxx pqt_data_source_dialog.h
  pqt_data_source_list_model.cxx pqt_data_source_list_model.h
  pqt_database.cxx pqt_database.h
  pqt_findscu.cxx pqt_findscu.h
  pqt_main.cxx
  pqt_main_window.cxx pqt_main_window.h
  pqt_patient_list_model.cxx pqt_patient_list_model.h
  )
if (QT4_FOUND AND QT_QTSQL_FOUND)
  qt4_wrap_cpp (PLASTIMATCH_QT_SRC 
    #    pqt_application.h
    pqt_data_source_dialog.h
    pqt_data_source_list_model.h
    pqt_main_window.h 
    pqt_patient_list_model.h
    )
  qt4_wrap_ui (PLASTIMATCH_QT_SRC 
    pqt_data_source_dialog.ui
    pqt_main_window.ui
    )
endif ()
set (CRYSTALVIEW_SRC
  cview_portal.cxx cview_portal.h
  cview_main.cxx cview_main.h
  )
if (QT4_FOUND)
  qt4_wrap_cpp (CRYSTALVIEW_SRC
    cview_portal.h
    cview_main.h
    )
endif ()
set (PROTON_DOSE_SRC
  proton_dose_main.cxx 
  proton_dose_opts.cxx proton_dose_opts.h
  )
set (LANDMARK_WARP_SRC
  landmark_warp_main.cxx
  )
set (LANDMARK_DIFF_SRC
  landmark_diff_main.cxx
  )
set (RANSAC_TEST_SRC 
  ransac_test.cxx
  )
set (RTOG_TO_MHA_SRC
  exchkeys.cxx exchkeys.h
  rtog_to_mha.cxx
  )
set (SHUFFLE_MHA_SRC
  shuffle_mha_main.cxx
  )
set (SIFTnD_SRC
  siftnD.cxx
  )
set (SYNTHETIC_VF_SRC
  synthetic_vf_main.cxx 
  )
set (TEST_EPS_SRC
  test_eps.cxx
  )
set (TEST_LIMITS_SRC
  test_limits.cxx
  )
set (TEST_OVERFLOW_SRC
  test_overflow.cxx
  )
set (VMATRIX_SRC
  Vmatrix.c
  )
set (XYZINT_SRC
  xyzInt.c
  )
set (VF_INVERT_SRC
  vf_invert.cxx
  )
set (XF_TO_EMPIREFMT_SRC
  xf_to_empirefmt.cxx
  )

##-----------------------------------------------------------------------------
##  SOURCE FILE PROPERTIES
##-----------------------------------------------------------------------------
if (OPENMP_FOUND)
  set_source_files_properties (bspline.cxx
    PROPERTIES COMPILE_FLAGS ${OPENMP_FLAGS})
  set_source_files_properties (bspline_mi.cxx
    PROPERTIES COMPILE_FLAGS ${OPENMP_FLAGS})
  set_source_files_properties (bspline_mse.cxx
    PROPERTIES COMPILE_FLAGS ${OPENMP_FLAGS})
  set_source_files_properties (bspline_warp.cxx
    PROPERTIES COMPILE_FLAGS ${OPENMP_FLAGS})
  set_source_files_properties (drr.cxx
    PROPERTIES COMPILE_FLAGS ${OPENMP_FLAGS})
  set_source_files_properties (fdk.cxx
    PROPERTIES COMPILE_FLAGS ${OPENMP_FLAGS})
  set_source_files_properties (bspline_regularize_analytic.cxx
    PROPERTIES COMPILE_FLAGS ${OPENMP_FLAGS})
  set_source_files_properties (openmp_test.cxx
    PROPERTIES COMPILE_FLAGS ${OPENMP_FLAGS})
endif ()

if (SSE2_FOUND AND NOT BUILD_AGAINST_SLICER3)
  plm_set_sse2_flags (bspline.cxx bspline_mi.cxx bspline_mse.cxx bspline_warp.cxx)
endif ()

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
plm_add_library (
  plmsys
  "${PLMSYS_LIBRARY_SRC}" 
  "${PLMSYS_LIBRARY_DEPENDENCIES}"
  "")

if (F2C_LIBRARY AND UNIX)
  plm_add_library (f2c_helper "${F2C_HELPER_LIBRARY_SRC}" "" "")
endif ()

plm_add_library (
  gpuit 
  "${GPUIT_LIBRARY_SRC}" 
  "${GPUIT_LIBRARY_DEPENDENCIES}"
  "${GPUIT_LIB_LDFLAGS}")

# JAS 2010.11.23
# Because the plugins are loaded at runtime, CMake doesn't know
# that they are needed by everything linked against gpuit.  So, we tell it.
if (BUILD_SHARED_LIBS)      # << Dynamic loading
  if (CUDA_FOUND)
    add_dependencies (gpuit plmcuda)
  endif ()
  if (OPENCL_FOUND)
    add_dependencies (gpuit plmopencl)
  endif ()
endif ()

if (FULL_PLASTIMATCH_BUILD)
  # Despite claims to the contrary, you are not allowed to have a library
  # with the same name as the executable.  What happens is the executable 
  # tries to create a temporary library with the same name.  [MSVC 2005]
  # That is why the library is called plastimatch1 instead of plastimatch.
  set (PLASTIMATCH1_LIBRARY_DEPENDENCIES ${ITK_LIBRARIES} ${GPUIT_LIBRARIES})

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

  plm_add_library (
    plastimatch1 
    "${PLASTIMATCH1_LIBRARY_SRC}"
    "${PLASTIMATCH1_LIBRARY_DEPENDENCIES}"
    "")
endif ()

##-----------------------------------------------------------------------------
##  Executable targets
##-----------------------------------------------------------------------------
if (NOT BUILD_DEBIAN)
  plm_add_executable (bragg_curve "${BRAGG_CURVE_SRC}" 
    "${GPUIT_LIBRARIES}" "${GPUIT_EXE_LDFLAGS}" 
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
plm_add_executable (drr "${DRR_SRC}" 
  "${GPUIT_LIBRARIES}" "${PLMCUDA_EXE_LDFLAGS}" 
  ${BUILD_IF_NOT_SLICER_EXT} ${INSTALL_ALWAYS})
plm_add_executable (dlib_train "${DLIB_TRAIN_SRC}" 
  "${GPUIT_LIBRARIES}" "${PLMCUDA_EXE_LDFLAGS}" 
  ${BUILD_IF_NOT_SLICER_EXT} ${INSTALL_IF_NOT_DEBIAN})
plm_add_executable (fdk "${FDK_SRC}" 
  "${GPUIT_LIBRARIES}" "${PLMCUDA_EXE_LDFLAGS}" 
  ${BUILD_IF_NOT_SLICER_EXT} ${INSTALL_ALWAYS})
plm_add_executable (reg "${REG_SRC}" 
  "${GPUIT_LIBRARIES}" "${GPUIT_EXE_LDFLAGS}" 
  ${BUILD_IF_NOT_SLICER_EXT} ${INSTALL_NEVER})
plm_add_executable (hnd_to_pfm "${HND_TO_PFM_SRC}" 
  "${GPUIT_LIBRARIES}" "${GPUIT_EXE_LDFLAGS}" 
  ${BUILD_IF_NOT_SLICER_EXT} ${INSTALL_IF_NOT_DEBIAN})
plm_add_executable (mha_to_raw mha_to_raw.c "" "" 
  ${BUILD_IF_NOT_SLICER_EXT} ${INSTALL_IF_NOT_DEBIAN})
plm_add_executable (proton_dose "${PROTON_DOSE_SRC}" 
  "${GPUIT_LIBRARIES}" "${GPUIT_EXE_LDFLAGS}" 
  ${BUILD_IF_NOT_SLICER_EXT} ${INSTALL_NEVER})
plm_add_executable (raw_to_mha raw_to_mha.c "" "" 
  ${BUILD_IF_NOT_SLICER_EXT} ${INSTALL_IF_NOT_DEBIAN})
plm_add_executable (test_eps "${TEST_EPS_SRC}" "" "" 
  ${BUILD_IF_NOT_SLICER_EXT} ${INSTALL_IF_NOT_DEBIAN})
plm_add_executable (test_limits "${TEST_LIMITS_SRC}" "" "" 
  ${BUILD_IF_NOT_SLICER_EXT} ${INSTALL_IF_NOT_DEBIAN})
plm_add_executable (test_overflow "${TEST_OVERFLOW_SRC}" "" "" 
  ${BUILD_IF_NOT_SLICER_EXT} ${INSTALL_IF_NOT_DEBIAN})
plm_add_executable (Vmatrix "${VMATRIX_SRC}" "" "" 
  ${BUILD_IF_NOT_SLICER_EXT} ${INSTALL_IF_NOT_DEBIAN})
plm_add_executable (xyzInt "${XYZINT_SRC}" "" "" 
  ${BUILD_IF_NOT_SLICER_EXT} ${INSTALL_IF_NOT_DEBIAN})

if (FULL_PLASTIMATCH_BUILD)
  message (STATUS "PLASTIMATCH_LIBS: ${PLASTIMATCH_LIBS}")
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
  plm_add_executable (extract_contour "${EXTRACT_CONTOUR_SRC}"
    "${PLASTIMATCH_LIBS}" "${GPUIT_EXE_LDFLAGS}" 
    ${BUILD_IF_NOT_SLICER_EXT} ${INSTALL_NEVER})
  plm_add_executable (landmark_warp "${LANDMARK_WARP_SRC}" 
    "${PLASTIMATCH_LIBS}" "${GPUIT_EXE_LDFLAGS}" 
    ${BUILD_IF_NOT_SLICER_EXT} ${INSTALL_ALWAYS})
  plm_add_executable (landmark_diff "${LANDMARK_DIFF_SRC}" 
    "${PLASTIMATCH_LIBS}" "${GPUIT_EXE_LDFLAGS}" 
    ${BUILD_IF_NOT_SLICER_EXT} ${INSTALL_NEVER})
  plm_add_executable (merge_vfs merge_vector_fields.cxx
    "${PLASTIMATCH_LIBS}" "${GPUIT_EXE_LDFLAGS}" 
    ${BUILD_IF_NOT_SLICER_EXT} ${INSTALL_IF_NOT_DEBIAN})
  plm_add_executable (mha_to_rtog_dose "${MHA_TO_RTOG_DOSE_SRC}"
    "${PLASTIMATCH_LIBS}" "${GPUIT_EXE_LDFLAGS}" 
    ${BUILD_IF_NOT_SLICER_EXT} ${INSTALL_NEVER})
  plm_add_executable (plastimatch "${PLASTIMATCH_SRC}"
    "${PLASTIMATCH_LIBS}" "${PLMCUDA_EXE_LDFLAGS}" 
    ${BUILD_ALWAYS} ${INSTALL_ALWAYS})
  if (QT4_FOUND)
    set (CRYSTALVIEW_LIBRARIES ${PLASTIMATCH_LIBS} 
      ${QT_LIBRARIES} ${QT_QTSQL_LIBRARIES})
    plm_add_executable (cview "${CRYSTALVIEW_SRC}"
      "${CRYSTALVIEW_LIBRARIES}" "${GPUIT_EXE_LDFLAGS}"
      ${BUILD_IF_NOT_SLICER_EXT} ${INSTALL_NEVER})
  endif ()
  if (QT4_FOUND AND QT_QTSQL_FOUND)
    set (PLASTIMATCH_QT_LIBRARIES ${PLASTIMATCH_LIBS} 
      ${QT_LIBRARIES} ${QT_QTSQL_LIBRARIES})
    plm_add_executable (plastimatch_qt "${PLASTIMATCH_QT_SRC}"
      "${PLASTIMATCH_QT_LIBRARIES}" "${GPUIT_EXE_LDFLAGS}" 
      ${BUILD_IF_NOT_SLICER_EXT} ${INSTALL_NEVER})
  endif ()
  plm_add_executable (ransac_test "${RANSAC_TEST_SRC}" 
    "${PLASTIMATCH_LIBS}" "${GPUIT_EXE_LDFLAGS}" 
    ${BUILD_IF_NOT_SLICER_EXT} ${INSTALL_NEVER})
  plm_add_executable (rtog_to_mha "${RTOG_TO_MHA_SRC}"
    "${PLASTIMATCH_LIBS}" "${GPUIT_EXE_LDFLAGS}" 
    ${BUILD_IF_NOT_SLICER_EXT} ${INSTALL_NEVER})
  plm_add_executable (shuffle_mha "${SHUFFLE_MHA_SRC}"
    "${PLASTIMATCH_LIBS}" "${GPUIT_EXE_LDFLAGS}" 
    ${BUILD_IF_NOT_SLICER_EXT} ${INSTALL_IF_NOT_DEBIAN})
  if (PLM_BUILD_SIFTND)
    plm_add_executable (siftnD "${SIFTnD_SRC}"
      "${PLASTIMATCH_LIBS}" "${GPUIT_EXE_LDFLAGS}" 
      ${BUILD_IF_NOT_SLICER_EXT} ${INSTALL_IF_NOT_DEBIAN})
  endif ()
  plm_add_executable (synthetic_vf "${SYNTHETIC_VF_SRC}"
    "${PLASTIMATCH_LIBS}" "${GPUIT_EXE_LDFLAGS}" 
    ${BUILD_IF_NOT_SLICER_EXT} ${INSTALL_IF_NOT_DEBIAN})
  plm_add_executable (union_mask union_mask.cxx
    "${PLASTIMATCH_LIBS}" "${GPUIT_EXE_LDFLAGS}" 
    ${BUILD_IF_NOT_SLICER_EXT} ${INSTALL_IF_NOT_DEBIAN})
  plm_add_executable (vf_invert "${VF_INVERT_SRC}"
    "${PLASTIMATCH_LIBS}" "${GPUIT_EXE_LDFLAGS}" 
    ${BUILD_IF_NOT_SLICER_EXT} ${INSTALL_IF_NOT_DEBIAN})
  plm_add_executable (xf_to_EMPIREFMT "${XF_TO_EMPIREFMT_SRC}"
    "${PLASTIMATCH_LIBS}" "${GPUIT_EXE_LDFLAGS}" 
    ${BUILD_IF_NOT_SLICER_EXT} ${INSTALL_NEVER})
endif ()

# executables that require cuda
if (CUDA_FOUND)
  plm_add_executable (cuda_probe "${CUDA_PROBE_SRC}" 
    "${CUDA_LIBRARIES}" "" 
    ${BUILD_ALWAYS} ${INSTALL_IF_NOT_DEBIAN})
endif ()

# executables that require opencl
if (OPENCL_FOUND)
  plm_add_executable (opencl_probe "${OPENCL_PROBE_SRC}"
    "${GPUIT_LIBRARIES}" "${GPUIT_EXE_LDFLAGS}" 
    ${BUILD_ALWAYS} ${INSTALL_IF_NOT_DEBIAN})
endif ()

# dicom_uid requires dcmtk
if (DCMTK_FOUND)
  plm_add_executable (dicom_uid "${DICOM_UID_SRC}" 
    "${DCMTK_LIBRARIES}" "" 
    ${BUILD_IF_NOT_SLICER_EXT} ${INSTALL_IF_NOT_DEBIAN})
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

# matlab mex files
if (MATLAB_FOUND)
  mex_target (mex_drr 
    "${MEX_DRR_SRC}" "${GPUIT_LIBRARIES}" "${GPUIT_EXE_LDFLAGS}")
endif ()

##-----------------------------------------------------------------------------
##  Simple test files
##-----------------------------------------------------------------------------
# Test executable -- cuda
if (CUDA_FOUND AND PLM_BUILD_CUDA_TEST)
  cuda_compile (CUDA_TEST_WRAPPERS cuda_test.cu)
  set (CUDA_TEST_SRC ${CUDA_TEST_WRAPPERS} cuda_test.cu)
  message (STATUS "CUDA_TEST_SRC: ${CUDA_TEST_SRC}")
  plm_add_executable (cuda_test "${CUDA_TEST_SRC}" 
    "${CUDA_LIBRARIES}" "" 
    ${BUILD_ALWAYS} ${INSTALL_NEVER})
  set_target_properties (cuda_test PROPERTIES LINKER_LANGUAGE CXX)
endif ()

# Test executable -- dcmtk
if (DCMTK_FOUND AND DCMTK_VERSION_36 AND PLM_BUILD_DCMTK_TEST
    AND FULL_PLASTIMATCH_BUILD)
  set (DCMTK_TEST_SRC
    dcmtk_test.cxx)
  plm_add_executable (dcmtk_test "${DCMTK_TEST_SRC}"
    "${PLASTIMATCH_LIBS}" "${GPUIT_EXE_LDFLAGS}" 
    ${BUILD_ALWAYS} ${INSTALL_NEVER})
endif ()

# Test executable -- dlib
if (PLM_BUILD_DLIB_TEST)
  set (DLIB_TEST_SRC dlib_test.cxx)
  set (DLIB_TEST_LIBS "")
  plm_add_executable (dlib_test "${DLIB_TEST_SRC}" 
    "${DLIB_TEST_LIBS}" "" 
    ${BUILD_ALWAYS} ${INSTALL_NEVER})
endif ()

# Test executable -- fann
if (FANN_FOUND AND PLM_BUILD_FANN_TEST)
  set (FANN_TEST_SRC fann_test.c)
  set (FANN_TEST_LIBS 
    ${GPUIT_LIBRARIES} ${FANN_LIBRARIES})
  plm_add_executable (fann_test "${FANN_TEST_SRC}" 
    "${FANN_TEST_LIBS}" "${OPENMP_LDFLAGS}"
    ${BUILD_ALWAYS} ${INSTALL_NEVER})
endif ()

# Test executable -- gdcm
if (ITK_FOUND AND GDCM_VERSION_1 AND PLM_BUILD_GDCM1_TEST)
  set (GDCM1_TEST_SRC
    gdcm1_test.cxx)
  plm_add_executable (gdcm1_test "${GDCM1_TEST_SRC}"
    "${PLASTIMATCH_LIBS}" "${GPUIT_EXE_LDFLAGS}" 
    ${BUILD_ALWAYS} ${INSTALL_NEVER})
endif ()

# Test executable -- mex
if (PLM_BUILD_MEX_TEST)
  if (OCTAVE_FOUND)
    add_custom_command (
      OUTPUT "${CMAKE_BINARY_DIR}/mex_test.mex"
      COMMAND ${OCTAVE_MKOCTFILE} --mex 
      -o "${CMAKE_BINARY_DIR}/mex_test.mex" 
      "${CMAKE_SOURCE_DIR}/mex_test.c"
      DEPENDS "${CMAKE_SOURCE_DIR}/mex_test.c")
    add_custom_target (oct_mex_test
      DEPENDS "${CMAKE_BINARY_DIR}/mex_test.mex")
  endif ()
  if (MATLAB_FOUND)
    file (WRITE "${CMAKE_BINARY_DIR}/compile_mex_test.m"
      "mex \"${CMAKE_SOURCE_DIR}/mex_test.c\";exit;\n")
    file (WRITE "${CMAKE_BINARY_DIR}/mex_test.cmake"
      "EXECUTE_PROCESS (COMMAND ${MATLAB_EXE} -nosplash -nodesktop -nojvm
       -r compile_mex_test
       RESULT_VARIABLE RESULT
       OUTPUT_VARIABLE STDOUT
       ERROR_VARIABLE STDERR)\n")
    add_custom_command (
      OUTPUT "${CMAKE_BINARY_DIR}/mex_test${MATLAB_MEXEXT}"
      COMMAND ${CMAKE_COMMAND} -P "${CMAKE_BINARY_DIR}/mex_test.cmake"
      DEPENDS "${CMAKE_SOURCE_DIR}/mex_test.c")
    add_custom_target (mat_mex_test
      DEPENDS "${CMAKE_BINARY_DIR}/mex_test${MATLAB_MEXEXT}")
    target_link_libraries (mex_test ${MATLAB_LIBRARIES})
  endif ()
endif ()

# Test executable -- nlopt
if (PLM_BUILD_NLOPT_TEST AND NLOPT_FOUND)
  set (NLOPT_TEST_SRC nlopt_test.c)
  set (NLOPT_TEST_LIBS 
    ${GPUIT_LIBRARIES} ${NLOPT_LIBRARIES})
  plm_add_executable (nlopt_test "${NLOPT_TEST_SRC}" 
    "${NLOPT_TEST_LIBS}" "${OPENMP_LDFLAGS}" 
    ${BUILD_ALWAYS} ${INSTALL_NEVER})
  set_target_properties (nlopt_test PROPERTIES LINKER_LANGUAGE CXX)
endif ()

# Test executable -- opencl
if (PLM_BUILD_OPENCL_TEST AND OPENCL_FOUND)
  set (OPENCL_TEST_SRC opencl_test.cxx opencl_test.h opencl_test.cl
    "${CMAKE_BINARY_DIR}/opencl_test.cl"
    )
  set (OPENCL_TEST_LIBS ${OPENCL_LIBRARIES} ${PLASTIMATCH_LIBS})
  plm_add_executable (opencl_test "${OPENCL_TEST_SRC}" 
    "${OPENCL_TEST_LIBS}" "" 
    ${BUILD_ALWAYS} ${INSTALL_NEVER})

  # I don't yet know how to bundle the .cl file within the executable.
  # Therefore, copy the .cl into binary directory.
  add_custom_command (
    OUTPUT "${CMAKE_BINARY_DIR}/opencl_test.cl" 
    COMMAND ${CMAKE_COMMAND} "-E" "copy" 
    "${CMAKE_CURRENT_SOURCE_DIR}/opencl_test.cl" 
    "${CMAKE_BINARY_DIR}/opencl_test.cl" 
    DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/opencl_test.cl")
endif ()

# Test executable -- openmp
if (PLM_BUILD_OPENMP_TEST AND OPENMP_FOUND)
  set (OPENMP_TEST_SRC openmp_test.c)
  plm_add_executable (openmp_test "${OPENMP_TEST_SRC}" 
    "${GPUIT_LIBRARIES}" "${OPENMP_LDFLAGS}" 
    ${BUILD_ALWAYS} ${INSTALL_NEVER})
endif ()

# Test executable -- cuda_tex_test
if (PLM_BUILD_CUDA_TEST AND CUDA_FOUND)
  add_subdirectory (CUDA_tex_test)
endif ()

# Test executable -- qt
if (PLM_BUILD_QT_TEST AND QT4_FOUND)
  set (QT_TEST_SRC qt_test.cxx)
  plm_add_executable (qt_test "${QT_TEST_SRC}" "${QT_LIBRARIES}" "" 
    ${BUILD_ALWAYS} ${INSTALL_NEVER})
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
    set (PLUGIN_DIR "${CMAKE_BINARY_DIR}/lib/Slicer-4.0/cli-modules")
    set (QTPLUGIN_DIR "${CMAKE_BINARY_DIR}/lib/Slicer-4.0/qt-loadable-modules")
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

    plm_slicer_copy_dll (plastimatch_slicer_copy_plastimatch1
      "${INDIR}/plastimatch1.dll"
      "${OUTDIR}/plastimatch1.dll"
      "${QTOUTDIR}/plastimatch1.dll"
      plastimatch1
      )
    plm_slicer_copy_dll (plastimatch_slicer_copy_gpuit
      "${INDIR}/gpuit.dll"
      "${OUTDIR}/gpuit.dll"
      "${QTOUTDIR}/gpuit.dll"
      gpuit
      )
    plm_slicer_copy_dll (plastimatch_slicer_copy_plmsys
      "${INDIR}/plmsys.dll"
      "${OUTDIR}/plmsys.dll"
      "${QTOUTDIR}/plmsys.dll"
      plmsys
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

## -------------------------------------------------------------------------
## Unit test targets
## -------------------------------------------------------------------------
if (ITK_FOUND)
  plm_add_executable (plm_drr_api_a plm_drr_api_a.cxx
    "${PLASTIMATCH_LIBS}" "${GPUIT_EXE_LDFLAGS}" 
    ${BUILD_IF_NOT_SLICER_EXT} ${INSTALL_NEVER})
  plm_add_executable (plm_registration_api_a plm_registration_api_a.cxx
    "${PLASTIMATCH_LIBS}" "${GPUIT_EXE_LDFLAGS}" 
    ${BUILD_IF_NOT_SLICER_EXT} ${INSTALL_NEVER})
endif ()
