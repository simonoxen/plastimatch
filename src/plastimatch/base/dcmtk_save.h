/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _dcmtk_save_h_
#define _dcmtk_save_h_

#include "plmbase_config.h"

class Rtds;

void
dcmtk_rtds_save (Rtds *rtds, const char *dicom_dir);

#endif
