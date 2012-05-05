/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _dcmtk_rtss_h_
#define _dcmtk_rtss_h_

#include "plmbase_config.h"

class Dcmtk_study_writer;
class Rtds;

void
dcmtk_rtss_save (
    Dcmtk_study_writer *dsw, 
    const Rtds *rtds,
    const char *dicom_dir);

#endif
