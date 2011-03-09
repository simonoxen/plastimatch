/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _itk_dicom_h_
#define _itk_dicom_h_

#include "plm_config.h"
#include "itk_image.h"
#include "plm_image_patient_position.h"

class Img_metadata;
class Referenced_dicom_dir;

void
itk_dicom_save (
    ShortImageType::Pointer short_img,      /* Input: image to write */
    const char *dir_name,                   /* Input: name of output dir */
    Referenced_dicom_dir *rdd,              /* Output: gets filled in */
    const Img_metadata *meta,               /* Input: output files get these */
    Plm_image_patient_position patient_pos  /* To be removed */
);

#endif
