/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _dcmtk_rtss_h_
#define _dcmtk_rtss_h_

#include "plm_config.h"
#include "dcmtk_series.h"

void
dcmtk_rtss_save (
    const std::vector<Dcmtk_slice_data> *slice_data,
    const Rtds *rtds,
    const char *dicom_dir);

#endif
