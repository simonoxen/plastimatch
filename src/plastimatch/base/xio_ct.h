/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _xio_ct_h_
#define _xio_ct_h_

#include "plmbase_config.h"

class Plm_image;
class Slice_index;
class Xio_studyset;

class Xio_ct_transform {
public:
    float direction_cosines[9];
    float x_offset;
    float y_offset;
};

PLMBASE_API void xio_ct_load (Plm_image *plm, const Xio_studyset *xio_studyset);
PLMBASE_API void xio_ct_get_transform_from_rdd (
    Plm_image *plm,
    Metadata *meta,
    Slice_index *rdd,
    Xio_ct_transform *transform
);
PLMBASE_API void xio_ct_get_transform ( Metadata *meta,
    Xio_ct_transform *transform
);
PLMBASE_API void xio_ct_apply_transform (Plm_image *plm, Xio_ct_transform *transform);

#endif
