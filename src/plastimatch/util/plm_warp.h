/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _plm_warp_h_
#define _plm_warp_h_

#include "plmutil_config.h"
#include "itkBSplineDeformableTransform.h"

class Plm_image;
class Xform;

/* -----------------------------------------------------------------------
   Public functions
   ----------------------------------------------------------------------- */
API void plm_warp (
    Plm_image *im_warped,   /* Output: Output image */
    DeformationFieldType::Pointer *vf,    /* Output: Output vf (optional) */
    Xform *xf_in,          /* Input:  Input image warped by this xform */
    Plm_image_header *pih,   /* Input:  Size of output image */
    Plm_image *im_in,       /* Input:  Input image */
    float default_val,     /* Input:  Value for pixels without match */
    int use_itk,           /* Input:  Force use of itk (1) or not (0) */
    int interp_lin         /* Input:  Trilinear (1) or nn (0) */
);

#endif
