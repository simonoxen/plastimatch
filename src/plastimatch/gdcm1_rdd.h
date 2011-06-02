/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _gdcm1_rdd_h_
#define _gdcm1_rdd_h_

#include "plm_config.h"
#if GDCM_VERSION_1

class Referenced_dicom_dir;

void
gdcm1_load_rdd (
    Referenced_dicom_dir *rdd,
    const char *dicom_dir
);

#endif /* GDCM_VERSION_1 */
#endif
