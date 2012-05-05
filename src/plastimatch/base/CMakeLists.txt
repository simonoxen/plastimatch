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
  astroid_dose.cxx                 #TODO
  bspline_xform.cxx
  cxt_io.cxx                       #TODO
  dcm_util.cxx                     #TODO
  dicom.dic
  exchkeys.cxx                     #TODO
  hnd_io.cxx                       #TODO
  interpolate.cxx
  itkClampCastImageFilter.txx      #TODO
  itk_dicom_load.cxx               #TODO
  itk_dicom_save.cxx               #TODO
  itk_directions.cxx               #TODO
  itk_image.cxx                    #TODO
  itk_image_cast.cxx               #TODO
  itk_image_load.txx               #TODO
  itk_image_load_char.cxx          #TODO
  itk_image_load_double.cxx        #TODO
  itk_image_load_float.cxx         #TODO
  itk_image_load_int32.cxx         #TODO
  itk_image_load_short.cxx         #TODO
  itk_image_load_uchar.cxx         #TODO
  itk_image_load_uint32.cxx        #TODO
  itk_image_load_ushort.cxx        #TODO
  itk_image_load_vec.cxx           #TODO
  itk_image_save.cxx               #TODO
  itk_image_stats.cxx              #TODO
  itk_pointset.cxx                 #TODO
  itk_resample.cxx                 #TODO
  itk_metadata.cxx                 #TODO
  itk_volume_header.cxx            #TODO
  mc_dose.cxx                      #TODO
  metadata.cxx                     #TODO
  mha_io.cxx
  plm_file_format.cxx              #TODO
  plm_image.cxx                    #TODO
  plm_image_convert.cxx            #TODO
  plm_image_header.cxx             #TODO
  plm_image_type.cxx               #TODO
  pointset.cxx
  raw_pointset.cxx
  ray_trace_exact.cxx
  ray_trace_uniform.cxx
  rpl_volume.cxx
  rtss_polyline_set.cxx            #TODO
  rtss_structure.cxx               #TODO
  slice_index.cxx                  #TODO
  ss_list_io.cxx                   #TODO
  thumbnail.cxx                    #TODO
  vf_convolve.cxx
  vf_stats.cxx                     #TODO
  vf.cxx
  volume_header.cxx                #TODO
  volume_limit.cxx
  volume_resample.cxx
  volume.cxx
  xform.cxx
  xform_convert.cxx                #TODO
  xform_legacy.cxx
  xform_point.cxx                  #TODO
  xio_ct.cxx                       #TODO
  xio_demographic.cxx              #TODO
  xio_dir.cxx                      #TODO
  xio_dose.cxx                     #TODO
  xio_patient.cxx                  #TODO
  xio_plan.cxx                     #TODO
  xio_structures.cxx               #TODO
  xio_studyset.cxx                 #TODO
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
  set (PLASTIMATCH1_LIBRARY_SRC
    ${PLASTIMATCH1_LIBRARY_SRC}
    gdcm2_util.cxx gdcm2_util.h
    )
endif ()

# JAS 2012.04.28 -- Commented out for Plastimatch 2.0 transition
#if (DCMTK_FOUND AND NOT DCMTK_VERSION_STRING VERSION_LESS 3.6.0)
#  set (PLMBASE_LIBRARY_SRC
#    ${PLMBASE_LIBRARY_SRC}
#    dcmtk_file.cxx
#    dcmtk_image.cxx
#    dcmtk_load.cxx
#    dcmtk_metadata.cxx
#    dcmtk_save.cxx
#    dcmtk_series.cxx
#    dcmtk_rtdose.cxx 
#    dcmtk_rtss.cxx
#    dcmtk_series_set.cxx
#    dcmtk_uid.cxx
#    dcmtk_util.cxx
#    )
#endif ()

##-----------------------------------------------------------------------------
##  LIBRARY DEPENDENCIES
##-----------------------------------------------------------------------------
set (PLMBASE_LIBRARY_DEPENDENCIES
  ${ITK_LIBRARIES}
  plmsys
  )

##-----------------------------------------------------------------------------
##  BUILD TARGETS
##-----------------------------------------------------------------------------
plm_add_library (
  plmbase
  "${PLMBASE_LIBRARY_SRC}" 
  "${PLMBASE_LIBRARY_DEPENDENCIES}"
  "")

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
