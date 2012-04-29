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
# JAS 2012.04.25 !!!!!!!!!!! TEMPORARY !!!!!!!!!!!!!!!!!!!!!!
include_directories (BEFORE ${CMAKE_CURRENT_SOURCE_DIR}/base)
include_directories (BEFORE ${CMAKE_CURRENT_SOURCE_DIR}/cli)
include_directories (BEFORE ${CMAKE_CURRENT_SOURCE_DIR}/cuda)
include_directories (BEFORE ${CMAKE_CURRENT_SOURCE_DIR}/register)
include_directories (BEFORE ${CMAKE_CURRENT_SOURCE_DIR}/script)
include_directories (BEFORE ${CMAKE_CURRENT_SOURCE_DIR}/segment)
include_directories (BEFORE ${CMAKE_CURRENT_SOURCE_DIR}/standalone)
include_directories (BEFORE ${CMAKE_CURRENT_SOURCE_DIR}/sys)
include_directories (BEFORE ${CMAKE_CURRENT_SOURCE_DIR}/util)
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
include_directories (BEFORE ${CMAKE_BINARY_DIR})
include_directories (BEFORE ${CMAKE_CURRENT_BINARY_DIR})
include_directories (AFTER ${BSTRLIB_INCLUDE_DIR})
include_directories (AFTER ${LIBLBFGS_INCLUDE_DIR})
include_directories (AFTER ${MSINTTYPES_INCLUDE_DIR})
if (NLOPT_FOUND)
  include_directories (AFTER ${NLOPT_INCLUDE_DIR})
endif ()
if (SQLITE_FOUND)
  include_directories (AFTER ${SQLITE_INCLUDE_DIR})
endif ()
include_directories (AFTER ${CMAKE_SOURCE_DIR}/libs/itk-3.20.0)
include_directories (AFTER ${CMAKE_SOURCE_DIR}/libs/nSIFT)
include_directories (AFTER ${LUA_INCLUDE_DIR})


# Don't do the below.  A user-built f2c directory contains ctype.h, which 
# conflicts with the C/C++ header of the same name.
#IF (PLM_USE_F2C)
#  INCLUDE_DIRECTORIES (BEFORE ${F2C_INCLUDE_DIR})
#ENDIF ()

if (CUDA_FOUND)
  include_directories (AFTER ${CUDA_INCLUDE_DIRS})
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
set (GPUIT_LIBRARY_SRC
  bowtie_correction.cxx bowtie_correction.h
  bspline_optimize_nlopt.cxx bspline_optimize_nlopt.h
  drr.cxx drr.h
  drr_trilin.cxx drr_trilin.h
  fdk.cxx fdk.h
  fdk_util.cxx fdk_util.h
  proj_image.cxx proj_image.h
  proj_image_dir.cxx proj_image_dir.h
  proj_matrix.cxx proj_matrix.h 
  proton_dose.cxx proton_dose.h 
  threading.cxx threading.h
  )

  # JAS 2012.4.27
  #   Temporary hack until CUDA linkage gets
  #      sorted out.
  if (CUDA_FOUND)
    set (GPUIT_CUDA_HACK
      cuda/cuda_util.cu
      drr_cuda.cu
      fdk_cuda.cu
      )
    cuda_compile (CUDA_WRAPPERS ${GPUIT_CUDA_HACK})
    set (GPUIT_LIBRARY_SRC
        ${GPUIT_LIBRARY_SRC}
        ${CUDA_WRAPPERS}
        drr_cuda.cpp
        fdk_cuda.cpp
    )
    set (GPUIT_LIBRARY_DEPENDENCIES
      ${GPUIT_LIBRARY_DEPENDENCIES}
      ${CUDA_LIBRARIES}
    )
  endif ()

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

if (QT4_FOUND)
  qt4_wrap_cpp (PORTAL_MOC_SRC
      cview_portal.h
  )
  qt4_wrap_cpp (CRYSTALVIEW_MOC_SRC
    cview_main.h
    )
endif ()

if (FFTW_FOUND)
  set (GPUIT_LIBRARY_SRC
    ${GPUIT_LIBRARY_SRC}
    ramp_filter.cxx ramp_filter.h
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
  autolabel.cxx autolabel.h 
  autolabel_task.cxx autolabel_task.h 
  autolabel_thumbnailer.cxx autolabel_thumbnailer.h 
  autolabel_trainer.cxx autolabel_trainer.h 
  autolabel_ransac_est.cxx autolabel_ransac_est.h 
  cxt_extract.cxx cxt_extract.h
  cxt_io.cxx cxt_io.h 
  dlib_trainer.cxx dlib_trainer.h 
  dvh.cxx dvh.h
  gamma_analysis.cxx gamma_analysis.h
  itk_crop.cxx itk_crop.h 
  itk_gabor.cxx itk_gabor.h
  itk_mask.cxx itk_mask.h 
  itk_tps.cxx itk_tps.h 
  landmark_warp.cxx landmark_warp.h 
  landmark_diff.cxx landmark_diff.h
  plm_clp.cxx plm_clp.h
  plm_drr_api.cxx plm_drr_api.h 
  plm_register_loadable.cxx plm_register_loadable.h 
  plm_registration_api.cxx plm_registration_api.h 
  rasterizer.cxx rasterizer.h
  rasterize_slice.cxx rasterize_slice.h
  rtds.cxx rtds.h
  rtds_dcmtk.cxx
  rtds_gdcm.cxx
  rtds_warp.cxx rtds_warp.h
  rtss.cxx rtss.h
  simplify_points.cxx simplify_points.h
  slice_extract.cxx slice_extract.h
  ss_img_stats.cxx ss_img_stats.h 
  ss_list_io.cxx ss_list_io.h
  synthetic_mha.cxx synthetic_mha.h
  synthetic_vf.cxx synthetic_vf.h
  threshbox.cxx threshbox.h
  )

if (ITK_FOUND AND GDCM_VERSION_2)
  set (PLASTIMATCH1_LIBRARY_SRC
    ${PLASTIMATCH1_LIBRARY_SRC}
    gdcm2_util.cxx gdcm2_util.h
    )
endif ()


##-----------------------------------------------------------------------------
##  SOURCE FILES (EXECUTABLES)
##-----------------------------------------------------------------------------
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
set (EXTRACT_CONTOUR_SRC
  extract_contour.cxx
  )
set (GABOR_TEST_SRC 
  gabor_test.cxx
  )
set (MEX_DRR_SRC
  mex_drr.c
  )
set (MHA_TO_RTOG_DOSE_SRC
  mha_to_rtog_dose.cxx
  )
set (MONDOSHOT_SRC
  dcmtk_uid.cxx dcmtk_uid.h
  mondoshot_dicom.cpp mondoshot_dicom.h
  mondoshot_main.cpp mondoshot_main.h
  )
set (OPENCL_PROBE_SRC
  opencl_probe_main.c
  )
if (OPENCL_FOUND)
  plm_add_opencl_file (GPUIT_LIBRARY_SRC opencl_probe.cl)
endif ()
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
  ${CRYSTALVIEW_MOC_SRC}
  ${PORTAL_MOC_SRC}
  cview_portal.cxx cview_portal.h
  cview_main.cxx cview_main.h
  )
set (LANDMARK_DIFF_SRC
  landmark_diff_main.cxx
  )
set (RANSAC_TEST_SRC 
  ransac_test.cxx
  )
set (RTOG_TO_MHA_SRC
  rtog_to_mha.cxx
  )
set (SHUFFLE_MHA_SRC
  shuffle_mha_main.cxx
  )
set (SIFTnD_SRC
  siftnD.cxx
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
#if (BUILD_SHARED_LIBS)      # << Dynamic loading
#  if (CUDA_FOUND)
#    add_dependencies (gpuit plmcuda)
#  endif ()
#  if (OPENCL_FOUND)
#    add_dependencies (gpuit plmopencl)
#  endif ()
#endif ()

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

  plm_add_library (
    plastimatch1 
    "${PLASTIMATCH1_LIBRARY_SRC}"
    "${PLASTIMATCH1_LIBRARY_DEPENDENCIES}"
    "")
endif ()

##-----------------------------------------------------------------------------
##  Executable targets
##-----------------------------------------------------------------------------
if (FULL_PLASTIMATCH_BUILD)
  message (STATUS "PLASTIMATCH_LIBS: ${PLASTIMATCH_LIBS}")
  plm_add_executable (extract_contour "${EXTRACT_CONTOUR_SRC}"
    "${PLASTIMATCH_LIBS}" "${GPUIT_EXE_LDFLAGS}" 
    ${BUILD_IF_NOT_SLICER_EXT} ${INSTALL_NEVER})
  plm_add_executable (gabor_test "${GABOR_TEST_SRC}" 
    "${PLASTIMATCH_LIBS}" "${GPUIT_EXE_LDFLAGS}" 
    ${BUILD_IF_NOT_SLICER_EXT} ${INSTALL_NEVER})
  plm_add_executable (landmark_diff "${LANDMARK_DIFF_SRC}" 
    "${PLASTIMATCH_LIBS}" "${GPUIT_EXE_LDFLAGS}" 
    ${BUILD_IF_NOT_SLICER_EXT} ${INSTALL_NEVER})
  plm_add_executable (merge_vfs merge_vector_fields.cxx
    "${PLASTIMATCH_LIBS}" "${GPUIT_EXE_LDFLAGS}" 
    ${BUILD_IF_NOT_SLICER_EXT} ${INSTALL_IF_NOT_DEBIAN})
  plm_add_executable (mha_to_rtog_dose "${MHA_TO_RTOG_DOSE_SRC}"
    "${PLASTIMATCH_LIBS}" "${GPUIT_EXE_LDFLAGS}" 
    ${BUILD_IF_NOT_SLICER_EXT} ${INSTALL_NEVER})
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


# plastimatch libraries
add_subdirectory(base)
add_subdirectory(cli)
add_subdirectory(register)
add_subdirectory(segment)
add_subdirectory(standalone)
add_subdirectory(script)
add_subdirectory(sys)
add_subdirectory(util)
