/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _plm_warp_h_
#define _plm_warp_h_

#include "plm_config.h"

/* -----------------------------------------------------------------------
   Public functions
   ----------------------------------------------------------------------- */
plastimatch1_EXPORT
void
plm_warp (
    PlmImage *im_warped,   /* Output: Output image */
    DeformationFieldType::Pointer *vf,    /* Output: Output vf (optional) */
    Xform *xf_in,          /* Input:  Input image warped by this xform */
    PlmImageHeader *pih,   /* Input:  Size of output image */
    PlmImage *im_in,       /* Input:  Input image */
    float default_val,     /* Input:  Value for pixels without match */
    int use_itk,           /* Input:  Force use of itk (1) or not (0) */
    int interp_lin         /* Input:  Trilinear (1) or nn (0) */
);

#endif
