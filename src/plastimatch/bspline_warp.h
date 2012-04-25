/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _bspline_warp_h_
#define _bspline_warp_h_

#include "plm_config.h"
#include "plmbase.h"

/* -----------------------------------------------------------------------
   Function declarations
   ----------------------------------------------------------------------- */
#if defined __cplusplus
extern "C" {
#endif

gpuit_EXPORT
void
bspline_warp (
    Volume *vout,         /* Output image (already sized and allocated) */
    Volume *vf_out,       /* Output vf (already sized and allocated, can be null) */
    Bspline_xform* bxf,   /* Bspline transform coefficients */
    Volume *moving,       /* Input image */
    int linear_interp,    /* 1 = trilinear, 0 = nearest neighbors */
    float default_val     /* Fill in this value outside of image */
);

#if defined __cplusplus
}
#endif

#endif
