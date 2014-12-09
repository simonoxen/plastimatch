/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _bspline_mi_txx_
#define _bspline_mi_txx_

void
bspline_mi_score_function (
    double& score_incr,       /* Output: score increase for this voxel */
    float dc_dv[3],           /* Output: dc_dv for this voxel */
    float m_val,              /* Input:  value in moving image */
    float f_val,              /* Input:  value in fixed image */
    float m_grad[3]           /* Input:  gradient in moving image */
)
{
    /* Compute intensity difference */
    float diff = m_val - f_val;
    score_incr = diff * diff;

    /* Compute spatial gradient using nearest neighbors */
    dc_dv[0] = diff * m_grad[0];  /* x component */
    dc_dv[1] = diff * m_grad[1];  /* y component */
    dc_dv[2] = diff * m_grad[2];  /* z component */
}

#endif
