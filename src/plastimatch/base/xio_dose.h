/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _xio_dose_h_
#define _xio_dose_h_

#include "plm_config.h"
#include "metadata.h"
#include "plm_image.h"
#include "xio_ct.h"

plastimatch1_EXPORT 
void
xio_dose_load (
    Plm_image *plm,
    Metadata* meta,
    const char *filename
);
plastimatch1_EXPORT 
void
xio_dose_save (
    Plm_image *plm,
    Metadata* meta,
    Xio_ct_transform *transform,
    const char *filename,
    const char *filename_template
);
plastimatch1_EXPORT 
void
xio_dose_apply_transform (Plm_image *plm, Xio_ct_transform *transform);

#endif
