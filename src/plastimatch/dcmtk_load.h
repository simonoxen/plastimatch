/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _dcmtk_load_h_
#define _dcmtk_load_h_

#include "plm_config.h"
#include "itk_image.h"

plastimatch1_EXPORT
ShortImageType::Pointer dcmtk_load (const char *dicom_dir);

#endif
