/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plm_config.h"

#include "dcmtk_load.h"

ShortImageType::Pointer 
dcmtk_load (const char *dicom_dir)
{
    ShortImageType::Pointer img = ShortImageType::New ();
    
    return img;
}
