/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _itk_dicom_h_
#define _itk_dicom_h_

#include "plm_config.h"
#include "itk_image.h"
#include "plm_image_patient_position.h"

class Img_metadata;

void
itk_dicom_save (
    ShortImageType::Pointer short_img, 
    const char* dir_name, 
    Img_metadata *img_metadata, 
    Plm_image_patient_position patient_pos);

#endif
