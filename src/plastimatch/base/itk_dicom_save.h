/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _itk_dicom_h_
#define _itk_dicom_h_

#include "plmbase_config.h"
#include "itk_image_type.h"

class Metadata;
class Slice_index;

void
itk_dicom_save (
    ShortImageType::Pointer short_img,  /* Input: image to write */
    const char *dir_name,               /* Input: name of output dir */
    Slice_index *rdd,                   /* Output: gets filled in */
    const Metadata *meta                /* Input: output files get these */
);

#endif
