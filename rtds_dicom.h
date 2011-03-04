/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _plm_dicom_h_
#define _plm_dicom_h_

#include "plm_config.h"
#include "rtds.h"

#if defined __cplusplus
extern "C" {
#endif

plastimatch1_EXPORT
void
rtds_patient_pos_from_dicom_dir (Rtds *rtds, const char *dicom_dir);

#if defined __cplusplus
}
#endif

#endif
