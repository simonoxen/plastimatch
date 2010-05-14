/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _xio_ct_h_
#define _xio_ct_h_

#include "plm_config.h"
#include "plm_image.h"

plastimatch1_EXPORT 
void
xio_ct_load (Plm_image *plm, char *input_dir);
plastimatch1_EXPORT 
void
xio_ct_apply_dicom_dir (Plm_image *plm, char *dicom_dir);

#endif
