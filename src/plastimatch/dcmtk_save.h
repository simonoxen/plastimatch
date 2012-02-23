/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _dcmtk_save_h_
#define _dcmtk_save_h_

#include "plm_config.h"

class Rtds;

void
dcmtk_save_rtds (Rtds *rtds, const char *dicom_dir);

#endif
