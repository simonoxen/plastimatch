/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _astroid_dose_h_
#define _astroid_dose_h_

#include "plmbase_config.h"

class Metadata;
class Plm_image;
class Xio_ct_transform;

C_API void astroid_dose_load (
        Plm_image *plm,
        Metadata *meta,
        const char *filename
);
C_API void astroid_dose_apply_transform (
        Plm_image *plm,
        Xio_ct_transform *transform
);

#endif
