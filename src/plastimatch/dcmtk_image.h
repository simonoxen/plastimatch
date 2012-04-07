/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _dcmtk_image_h_
#define _dcmtk_image_h_

#include "plm_config.h"
#include "dcmtk_series.h"

void
dcmtk_image_save (
    std::vector<Dcmtk_slice_data> *slice_data,
    Rtds *rtds, 
    const char *dicom_dir);

#endif
