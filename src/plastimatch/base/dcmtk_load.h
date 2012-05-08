/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _dcmtk_load_h_
#define _dcmtk_load_h_

#include "plmbase_config.h"
#include "itk_image_type.h"

API ShortImageType::Pointer dcmtk_load (const char *dicom_dir);

#endif
