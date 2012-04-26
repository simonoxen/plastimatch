##-----------------------------------------------------------------------------
##  See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
##-----------------------------------------------------------------------------
project (src_plastimatch_base)

configure_file (
    ${CMAKE_CURRENT_SOURCE_DIR}/plmbase_config.h.in
    ${PLM_BUILD_ROOT}/plmbase_config.h
)

include_directories (BEFORE ${CMAKE_CURRENT_SOURCE_DIR})

set (PLMBASE_LIBRARY_SRC
  astroid_dose.cxx astroid_dose.h
  bspline_xform.cxx bspline_xform.h
  dcm_util.cxx dcm_util.h
  dicom.dic
  direction_cosines.h
  exchkeys.cxx exchkeys.h
  hnd_io.cxx hnd_io.h
  interpolate.cxx
  itkAndConstantToImageFilter.h
  itkClampCastImageFilter.txx itkClampCastImageFilter.h
  itk_dicom_load.cxx itk_dicom_load.h 
  itk_dicom_save.cxx itk_dicom_save.h 
  itk_directions.cxx itk_directions.h 
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
  itk_image_type.h
  itk_point.h
  itk_pointset.cxx itk_pointset.h
  itk_resample.cxx itk_resample.h 
  itk_metadata.cxx itk_metadata.h 
  itk_volume_header.cxx itk_volume_header.h
  make_string.h
  mc_dose.cxx mc_dose.h
  metadata.cxx metadata.h 
  mha_io.cxx
  plm_file_format.cxx plm_file_format.h 
  plm_image.cxx plm_image.h 
  plm_image_convert.cxx plm_image_convert.h 
  plm_image_header.cxx plm_image_header.h 
  plm_image_type.cxx plm_image_type.h 
  plm_uid_prefix.h
  pointset.cxx pointset.h 
  raw_pointset.cxx raw_pointset.h 
  ray_trace_exact.cxx
  ray_trace_uniform.cxx
  rpl_volume.cxx rpl_volume.h
  rtss_polyline_set.cxx rtss_polyline_set.h 
  rtss_structure.cxx rtss_structure.h 
  slice_index.cxx slice_index.h
  thumbnail.cxx thumbnail.h
  vf_convolve.cxx
  vf_stats.cxx
  vf.cxx
  volume_header.cxx volume_header.h 
  volume_limit.cxx volume_limit.h
  volume_macros.h
  volume_resample.cxx volume_resample.h
  volume.cxx volume.h
  xform.cxx xform.h 
  xform_convert.cxx xform_convert.h 
  xform_legacy.cxx xform_legacy.h
  xform_point.cxx xform_point.h 
  xio_ct.cxx xio_ct.h
  xio_demographic.cxx xio_demographic.h
  xio_dir.cxx xio_dir.h
  xio_dose.cxx xio_dose.h
  xio_patient.cxx xio_patient.h 
  xio_plan.cxx xio_plan.h
  xio_structures.cxx xio_structures.h
  xio_studyset.cxx xio_studyset.h
  xpm.cxx xpm.h
  )
if (ITK_FOUND AND GDCM_VERSION_1)
  set (PLMBASE_LIBRARY_SRC
    ${PLMBASE_LIBRARY_SRC}
    gdcm1_dose.cxx gdcm1_dose.h
    gdcm1_file.cxx gdcm1_file.h
    gdcm1_rdd.cxx gdcm1_rdd.h 
    gdcm1_rtss.cxx gdcm1_rtss.h
    gdcm1_series.cxx gdcm1_series.h
    gdcm1_series_helper_2.cxx gdcm1_series_helper_2.h
    gdcm1_util.cxx gdcm1_util.h
    )
endif ()

plm_add_library (
  plmbase
  "${PLMBASE_LIBRARY_SRC}" 
  "${PLMBASE_LIBRARY_DEPENDENCIES}"
  "")

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
