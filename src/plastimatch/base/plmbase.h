/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _plmbase_h_
#define _plmbase_h_

#include "plmbase_config.h"

/* gdcm must be before headers including plm_int.h */
#if GDCM_VERSION_1
#include "gdcm1_dose.h"
#include "gdcm1_file.h"
#include "gdcm1_rdd.h"
#include "gdcm1_rtss.h"
#include "gdcm1_series.h"
#include "gdcm1_series_helper_2.h"
#include "gdcm1_util.h"
#endif
#if GDCM_VERSION_2
#include "gdcm2_util.h"
#endif

#include "astroid_dose.h"
#include "bspline_xform.h"
#include "cxt_io.h"
#include "dcm_util.h"
#include "hnd_io.h"
#include "interpolate.h"
#if (!PLM_CUDA_COMPILE)
#include "itkClampCastImageFilter.h"
#include "itk_dicom_load.h"
#include "itk_dicom_save.h"
#include "itk_directions.h"
#include "itk_image.h"
#include "itk_image_cast.h"
#include "itk_image_load.h"
#include "itk_image_save.h"
#include "itk_image_stats.h"
#include "itk_image_type.h"
#include "itk_metadata.h"
#include "itk_pointset.h"
#include "itk_resample.h"
#include "itk_volume_header.h"
#endif
#include "mc_dose.h"
#include "metadata.h"
#include "mha_io.h"
#include "plm_file_format.h"
#if (!PLM_CUDA_COMPILE)
#include "plm_image.h"
#include "plm_image_convert.h"
#include "plm_image_header.h"
#endif
#include "plm_image_type.h"
#include "pointset.h"
#include "raw_pointset.h"
#include "ray_trace.h"
#include "rpl_volume.h"
#include "rtss_polyline_set.h"
#include "rtss_structure.h"
#include "ss_list_io.h"
#if (!PLM_CUDA_COMPILE)
#include "slice_index.h"
#include "thumbnail.h"
#endif
#include "vf.h"
#include "vf_convolve.h"
#include "vf_stats.h"
#include "volume.h"
#include "volume_header.h"
#include "volume_limit.h"
#include "volume_resample.h"
#include "xpm.h"
#if (!PLM_CUDA_COMPILE)
#include "xform.h"
#include "xform_convert.h"
#include "xform_point.h"
#include "xform_legacy.h"
#endif
#include "xio_ct.h"
#include "xio_dir.h"
#include "xio_dose.h"
#include "xio_demographic.h"
#include "xio_patient.h"
#include "xio_plan.h"
#include "xio_structures.h"
#include "xio_studyset.h"

#endif
