/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _bspline_correspond_h_
#define _bspline_correspond_h_

class Volume;

API int inside_mask (float* xyz, const Volume* mask);

API int bspline_find_correspondence (
    float *mxyz,             /* Output: xyz coordinates in moving image (mm) */
    float *mijk,             /* Output: ijk indices in moving image (vox) */
    const float *fxyz,       /* Input:  xyz coordinates in fixed image (mm) */
    const float *dxyz,       /* Input:  displacement from fixed to moving (mm) */
    const Volume *moving     /* Input:  moving image */
);

API int bspline_find_correspondence_dcos (
    float *mxyz,             /* Output: xyz coordinates in moving image (mm) */
    float *mijk,             /* Output: ijk indices in moving image (vox) */
    const float *fxyz,       /* Input:  xyz coordinates in fixed image (mm) */
    const float *dxyz,       /* Input:  displacement from fixed to moving (mm) */
    const Volume *moving     /* Input:  moving image */
);

API int bspline_find_correspondence_dcos_mask (
    float *mxyz,               /* Output: xyz coordinates in moving image (mm) */
    float *mijk,               /* Output: ijk indices in moving image (vox) */
    const float *fxyz,         /* Input:  xyz coordinates in fixed image (mm) */
    const float *dxyz,         /* Input:  displacement from fixed to moving (mm) */
    const Volume *moving,      /* Input:  moving image */
    const Volume *moving_mask  /* Input:  moving image mask */
);

#endif