/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _dcmtk_image_h_
#define _dcmtk_image_h_

#include "plmbase_config.h"

class Dcmtk_study_writer;
class Rtds;

void
dcmtk_image_save (
    Dcmtk_study_writer *dsw, 
    Rtds *rtds, 
    const char *dicom_dir);

#endif
