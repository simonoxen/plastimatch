##-----------------------------------------------------------------------------
##  See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
##-----------------------------------------------------------------------------
project (src_plastimatch_base)

configure_file (
    ${CMAKE_CURRENT_SOURCE_DIR}/plmbase_config.h.in
    ${PLM_BUILD_ROOT}/plmbase_config.h
)

include_directories (BEFORE ${CMAKE_CURRENT_SOURCE_DIR})

##-----------------------------------------------------------------------------
##  SOURCE FILES
##-----------------------------------------------------------------------------
set (PLMBASE_LIBRARY_SRC
  astroid_dose.cxx
  bspline_interpolate.cxx
  bspline_xform.cxx
  cxt_io.cxx
  dicom_probe.cxx
  dicom_rt_study.cxx
  dicom_util.cxx
  dicom.dic
  exchkeys.cxx
  hnd_io.cxx
  interpolate.cxx
  itkClampCastImageFilter.txx
  itk_dicom_load.cxx
  itk_dicom_save.cxx
  itk_directions.cxx
  itk_image.cxx
  itk_image_cast.cxx
  itk_image_load.txx
  itk_image_load_char.cxx
  itk_image_load_double.cxx
  itk_image_load_float.cxx
  itk_image_load_int32.cxx
  itk_image_load_short.cxx
  itk_image_load_uchar.cxx
  itk_image_load_uint32.cxx
  itk_image_load_ushort.cxx
  itk_image_load_vec.cxx
  itk_image_save.cxx
  itk_image_stats.cxx
  itk_image_type.h
  itk_metadata.cxx
  itk_point.h
  itk_pointset.cxx
  itk_resample.cxx
  itk_volume_header.cxx
  mc_dose.cxx
  metadata.cxx
  mha_io.cxx
  plm_file_format.cxx
  plm_image.cxx
  plm_image_convert.cxx
  plm_image_header.cxx
  plm_image_type.cxx
  pointset.cxx
  proj_image.cxx
  proj_image_dir.cxx
  proj_matrix.cxx
  proj_volume.cxx proj_volume.h
  raw_pointset.cxx
  ray_trace_exact.cxx
  ray_trace_probe.cxx
  ray_trace_uniform.cxx
  rpl_volume.cxx
  rtog_io.cxx
  rtss_structure_set.cxx
  rtss_structure.cxx
  slice_index.cxx
  ss_list_io.cxx
  thumbnail.cxx
  vf_convolve.cxx
  vf_jacobian.cxx
  vf_stats.cxx
  vf.cxx
  volume_header.cxx
  volume_limit.cxx
  volume_resample.cxx
  volume_stats.cxx
  volume.cxx
  xform.cxx
  xform_convert.cxx
  xform_legacy.cxx
  xform_point.cxx
  xio_ct.cxx
  xio_demographic.cxx
  xio_dir.cxx
  xio_dose.cxx
  xio_patient.cxx
  xio_plan.cxx
  xio_structures.cxx
  xio_studyset.cxx
  xpm.cxx
  )

if (ITK_FOUND AND GDCM_VERSION_1)
  set (PLMBASE_LIBRARY_SRC
    ${PLMBASE_LIBRARY_SRC}
    gdcm1_dose.cxx
    gdcm1_file.cxx
    gdcm1_rdd.cxx
    gdcm1_rtss.cxx
    gdcm1_series.cxx
    gdcm1_series_helper_2.cxx
    gdcm1_util.cxx
    )
endif ()

if (ITK_FOUND AND GDCM_VERSION_2)
  set (PLMBASE_LIBRARY_SRC
    ${PLMBASE_LIBRARY_SRC}
    gdcm2_util.cxx
    )
endif ()

if (DCMTK_FOUND)
  set (PLMBASE_LIBRARY_SRC
    ${PLMBASE_LIBRARY_SRC}
    dcmtk_file.cxx
    dcmtk_image.cxx
    dcmtk_loader.cxx
    dcmtk_metadata.cxx
    dcmtk_rdd.cxx
    dcmtk_rtdose.cxx 
    dcmtk_rtss.cxx
    dcmtk_save.cxx
    dcmtk_series.cxx
    dcmtk_rt_study.cxx dcmtk_rt_study.h
    dcmtk_uid.cxx
    dcmtk_util.cxx
    )
endif ()

set (PLMBASE_LIBRARY_HEADERS
  "${CMAKE_BINARY_DIR}/plmbase_config.h"
  )
foreach (ARG ${PLMBASE_LIBRARY_SRC})
  string (REGEX MATCH ".*\\.h$" TMP "${ARG}")
  if (TMP)
    list (APPEND PLMBASE_LIBRARY_HEADERS "${TMP}")
  endif ()
endforeach ()

##-----------------------------------------------------------------------------
##  LIBRARY DEPENDENCIES
##-----------------------------------------------------------------------------
set (PLMBASE_LIBRARY_DEPENDENCIES
  ${ITK_LIBRARIES}
  plmsys
  )
if (DCMTK_FOUND)
  set (PLMBASE_LIBRARY_DEPENDENCIES
    ${PLMBASE_LIBRARY_DEPENDENCIES}
    ${DCMTK_LIBRARIES})
endif ()

##-----------------------------------------------------------------------------
##  BUILD TARGETS
##-----------------------------------------------------------------------------
plm_add_library (
  plmbase
  "${PLMBASE_LIBRARY_SRC}" 
  "${PLMBASE_LIBRARY_DEPENDENCIES}"
  ""
  "${PLMBASE_LIBRARY_HEADERS}")

##-----------------------------------------------------------------------------
##  SPECIAL RULES FOR SLICER
##-----------------------------------------------------------------------------
if (SLICER_FOUND)
  if (WIN32)
    plm_slicer_copy_dll (plastimatch_slicer_copy_plmbase
      "${INDIR}/plmbase.dll"
      "${OUTDIR}/plmbase.dll"
      "${QTOUTDIR}/plmbase.dll"
      plmbase
    )
  endif ()
endif ()
