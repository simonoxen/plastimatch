/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _dcmtk_rtss_h_
#define _dcmtk_rtss_h_

#include "plmbase_config.h"

class Dicom_rt_study;
class Rtds;

PLMBASE_C_API bool 
dcmtk_rtss_probe (const char *rtss_fn);

PLMBASE_C_API void
dcmtk_rtss_save (
    Dicom_rt_study *dsw, 
    const Rtds *rtds,
    const char *dicom_dir);

#endif
