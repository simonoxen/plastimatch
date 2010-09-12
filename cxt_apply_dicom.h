/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _cxt_apply_dicom_h_
#define _cxt_apply_dicom_h_

#include "plm_config.h"
#include "cxt_io.h"

#if defined __cplusplus
extern "C" {
#endif

plastimatch1_EXPORT
void
cxt_apply_dicom_dir (Rtss *cxt, const char *dicom_dir);

#if defined __cplusplus
}
#endif

#endif
