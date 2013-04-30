/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _gdcm1_rdd_h_
#define _gdcm1_rdd_h_

#include "plmbase_config.h"
#if GDCM_VERSION_1

class Rt_study_metadata;

void
gdcm1_load_rdd (
    Rt_study_metadata *rdd,
    const char *dicom_dir
);

#endif /* GDCM_VERSION_1 */
#endif
