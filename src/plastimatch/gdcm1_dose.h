/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _gdcm1_dose_h_
#define _gdcm1_dose_h_

#include "plm_config.h"

class Metadata;
class Plm_image;
class Slice_index;

#if defined __cplusplus
extern "C" {
#endif

plastimatch1_EXPORT
bool
gdcm1_dose_probe (const char *dose_fn);
plastimatch1_EXPORT
Plm_image*
gdcm1_dose_load (Plm_image *pli, const char *dose_fn, const char *dicom_dir);
plastimatch1_EXPORT
void
gdcm1_dose_save (
    Plm_image *pli, 
    const Metadata *meta, 
    const Slice_index *rdd, 
    const char *dose_fn);

#if defined __cplusplus
}
#endif

#endif
