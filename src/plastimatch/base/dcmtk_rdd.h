/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _dcmtk_rdd_h_
#define _dcmtk_rdd_h_

#include "plmbase_config.h"

class Slice_index;

void
dcmtk_load_rdd (
    Slice_index *rdd,
    const char *dicom_dir
);

#endif
