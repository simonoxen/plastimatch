/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _bspline_mse_txx_
#define _bspline_mse_txx_

void
bspline_mse_score_function_a (
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

void
bspline_mse_score_function_b (
    Bspline_optimize *bod,    /* In/out: generic optimization data */
    plm_long fidx,            /* Input:  index of voxel in fixed image */
    plm_long midx_f,          /* Input:  index (floor) in moving image*/
    plm_long mijk_r[3],       /* Input:  coords (rounded) in moving image*/
    plm_long pidx,            /* Input:  region index of fixed voxel */
    plm_long qidx,            /* Input:  offset index of fixed voxel */
    float li_1[3],            /* Input:  linear interpolation fraction */
    float li_2[3],            /* Input:  linear interpolation fraction */
    void *user_data           /* In/out: private function data */
)
{
    Bspline_parms *parms = bod->get_bspline_parms ();
    Bspline_state *bst = bod->get_bspline_state ();
    Bspline_xform *bxf = bod->get_bspline_xform ();
    Bspline_score *ssd = &bst->ssd;

    Volume *fixed = parms->fixed;
    Volume *moving = parms->moving;
    Volume *moving_grad = parms->moving_grad;
    float* f_img = (float*) fixed->img;
    float* m_img = (float*) moving->img;
    float* m_grad = (float*) moving_grad->img;

    float m_val;
    float dc_dv[3];

    double *score_acc = (double*) user_data;

    /* Get value in fixed image */
    float f_val = f_img[fidx];

    /* Compute moving image intensity using linear interpolation */
    /* Macro is slightly faster than function */
    LI_VALUE (m_val, 
        li_1[0], li_2[0],
        li_1[1], li_2[1],
        li_1[2], li_2[2],
        midx_f, m_img, moving);

    /* This replaces the commented out code */
    plm_long mvr = volume_index (moving->dim, mijk_r);

    /* Compute intensity difference */
    float diff = m_val - f_val;

    /* Update score */
    *score_acc += diff * diff;

    /* Compute spatial gradient using nearest neighbors */
    dc_dv[0] = diff * m_grad[0];  /* x component */
    dc_dv[1] = diff * m_grad[1];  /* y component */
    dc_dv[2] = diff * m_grad[2];  /* z component */

    /* Update cost function gradient */
    bspline_update_grad_b (&bst->ssd, bxf, pidx, qidx, dc_dv);
    ssd->num_vox++;
}

#endif
