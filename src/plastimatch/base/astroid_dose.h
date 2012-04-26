/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _astroid_dose_h_
#define _astroid_dose_h_

#include "plmbase_config.h"
#include "xio_ct.h"

class Metadata;
class Plm_image;

plastimatch1_EXPORT 
void
astroid_dose_load (
    Plm_image *plm,
    Metadata *meta,
    const char *filename
);
plastimatch1_EXPORT 
void
astroid_dose_apply_transform (Plm_image *plm, Xio_ct_transform *transform);

#endif
