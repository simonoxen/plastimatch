/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _gdcm_dose_h_
#define _gdcm_dose_h_

#include "plm_config.h"
#include "cxt_io.h"

#if defined __cplusplus
extern "C" {
#endif

plastimatch1_EXPORT
bool
gdcm_dose_probe (const char *dose_fn);
plastimatch1_EXPORT
Plm_image*
gdcm_dose_load (Plm_image *pli, const char *dose_fn, const char *dicom_dir);
plastimatch1_EXPORT
void
gdcm_dose_save (Plm_image *pli, char *dose_fn);

#if defined __cplusplus
}
#endif

#endif
