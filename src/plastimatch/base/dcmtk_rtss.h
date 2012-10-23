/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _dcmtk_rtss_h_
#define _dcmtk_rtss_h_

#include "plmbase_config.h"

class Dcmtk_rt_study;
class Rtds;

void
dcmtk_rtss_save (
    Dcmtk_rt_study *dsw, 
    const Rtds *rtds,
    const char *dicom_dir);

#endif
