/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _astroid_dose_h_
#define _astroid_dose_h_

#include "plm_config.h"
#include "plm_image.h"
#include "xio_ct.h"

plastimatch1_EXPORT 
void
astroid_dose_load (
    Plm_image *plm,
    Img_metadata *img_metadata,
    const char *filename
);
plastimatch1_EXPORT 
void
astroid_dose_apply_transform (Plm_image *plm, Xio_ct_transform *transform);

#endif
