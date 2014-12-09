/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _bspline_loop_txx_
#define _bspline_loop_txx_

#include "plmregister_config.h"
#include <string>
#include "bspline_macros.h"
#include "bspline_mse.h"
#include "bspline_optimize.h"
#include "bspline_parms.h"
#include "file_util.h"
#include "interpolate.h"
#include "interpolate_macros.h"
#include "plm_timer.h"
#include "string_util.h"

void
Bspline_mse_score_function (
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


template< void (*Bspline_score_function) (
    double& score_incr,       /* Output: score increase for this voxel */
    float dc_dv[3],           /* Output: dc_dv for this voxel */
    float m_val,              /* Input:  value in moving image */
    float f_val,              /* Input:  value in fixed image */
    float m_grad[3]           /* Input:  gradient in moving image */
) >
void
bspline_loop (Bspline_optimize *bod)
{
    Bspline_parms *parms = bod->get_bspline_parms ();
    Bspline_state *bst = bod->get_bspline_state ();
    Bspline_xform *bxf = bod->get_bspline_xform ();

    Volume *fixed = parms->fixed;
    Volume *moving = parms->moving;
    Volume *moving_grad = parms->moving_grad;

    Bspline_score* ssd = &bst->ssd;
    plm_long fijk[3], fv;         /* Indices within fixed image (vox) */
    float mijk[3];              /* Indices within moving image (vox) */
    float fxyz[3];              /* Position within fixed image (mm) */
    float mxyz[3];              /* Position within moving image (mm) */
    plm_long mijk_f[3], mvf;      /* Floor */
    plm_long mijk_r[3], mvr;      /* Round */
    plm_long p[3], pidx;          /* Region index of fixed voxel */
    plm_long q[3], qidx;          /* Offset index of fixed voxel */

    float dc_dv[3];
    float li_1[3];           /* Fraction of interpolant in lower index */
    float li_2[3];           /* Fraction of interpolant in upper index */
    float* f_img = (float*) fixed->img;
    float* m_img = (float*) moving->img;
    float* m_grad = (float*) moving_grad->img;
    float dxyz[3];
    float m_val;

    /* GCS: Oct 5, 2009.  We have determined that sequential accumulation
       of the score requires double precision.  However, reduction 
       accumulation does not. */
    double score_acc = 0.;

    static int it = 0;
    FILE* val_fp = 0;
    FILE* dc_dv_fp = 0;
    FILE* corr_fp = 0;

    if (parms->debug) {
        std::string fn;

        fn = string_format ("%s/%02d_dc_dv_mse_%03d_%03d.csv",
            parms->debug_dir.c_str(), parms->debug_stage, bst->it, 
            bst->feval);
        dc_dv_fp = plm_fopen (fn.c_str(), "wb");

        fn = string_format ("%s/%02d_val_mse_%03d_%03d.csv",
            parms->debug_dir.c_str(), parms->debug_stage, bst->it, 
            bst->feval);
        val_fp = plm_fopen (fn.c_str(), "wb");

        fn = string_format ("%s/%02d_corr_mse_%03d_%03d.csv",
            parms->debug_dir.c_str(), parms->debug_stage, bst->it, 
            bst->feval);
        corr_fp = plm_fopen (fn.c_str(), "wb");
        it ++;
    }

    Plm_timer* timer = new Plm_timer;
    timer->start ();

    /* GCS FIX: region of interest is not used */
    LOOP_Z (fijk, fxyz, fixed) {
        p[2] = REGION_INDEX_Z (fijk, bxf);
        q[2] = REGION_OFFSET_Z (fijk, bxf);
        LOOP_Y (fijk, fxyz, fixed) {
            p[1] = REGION_INDEX_Y (fijk, bxf);
            q[1] = REGION_OFFSET_Y (fijk, bxf);
            LOOP_X (fijk, fxyz, fixed) {
                p[0] = REGION_INDEX_X (fijk, bxf);
                q[0] = REGION_OFFSET_X (fijk, bxf);

                /* Get B-spline deformation vector */
                pidx = volume_index (bxf->rdims, p);
                qidx = volume_index (bxf->vox_per_rgn, q);
                bspline_interp_pix_b (dxyz, bxf, pidx, qidx);

                /* Compute moving image coordinate of fixed image voxel */
                mxyz[2] = fxyz[2] + dxyz[2] - moving->offset[2];
                mxyz[1] = fxyz[1] + dxyz[1] - moving->offset[1];
                mxyz[0] = fxyz[0] + dxyz[0] - moving->offset[0];
                mijk[2] = PROJECT_Z (mxyz, moving->proj);
                mijk[1] = PROJECT_Y (mxyz, moving->proj);
                mijk[0] = PROJECT_X (mxyz, moving->proj);

                if (parms->debug) {
                    fprintf (corr_fp, 
                        "%d %d %d, %f %f %f -> %f %f %f, %f %f %f\n",
                        (unsigned int) fijk[0], 
                        (unsigned int) fijk[1], 
                        (unsigned int) fijk[2], 
                        fxyz[0], fxyz[1], fxyz[2],
                        mijk[0], mijk[1], mijk[2],
                        fxyz[0] + dxyz[0], fxyz[1] + dxyz[1], fxyz[2] + dxyz[2]
                    );
                }

                if (mijk[2] < -0.5 || mijk[2] > moving->dim[2] - 0.5) continue;
                if (mijk[1] < -0.5 || mijk[1] > moving->dim[1] - 0.5) continue;
                if (mijk[0] < -0.5 || mijk[0] > moving->dim[0] - 0.5) continue;

                /* Compute interpolation fractions */
                li_clamp_3d (mijk, mijk_f, mijk_r, li_1, li_2, moving);

                /* Find linear index of "corner voxel" in moving image */
                mvf = volume_index (moving->dim, mijk_f);

                /* Compute moving image intensity using linear interpolation */
                /* Macro is slightly faster than function */
                LI_VALUE (m_val, 
                    li_1[0], li_2[0],
                    li_1[1], li_2[1],
                    li_1[2], li_2[2],
                    mvf, m_img, moving);

                /* Compute linear index of fixed image voxel */
                fv = volume_index (fixed->dim, fijk);

                /* This replaces the commented out code */
                double score_incr;
                mvr = volume_index (moving->dim, mijk_r);
                Bspline_mse_score_function (
                    score_incr,
                    dc_dv,
                    m_val,
                    f_img[fv],
                    &m_grad[3*mvr]);
                bspline_update_grad_b (&bst->ssd, bxf, pidx, qidx, dc_dv);
                score_acc += score_incr;
                ssd->num_vox++;

#if defined (commentout)
                /* Compute intensity difference */
                float diff = m_val - f_img[fv];

                /* Compute spatial gradient using nearest neighbors */
                mvr = volume_index (moving->dim, mijk_r);
                dc_dv[0] = diff * m_grad[3*mvr+0];  /* x component */
                dc_dv[1] = diff * m_grad[3*mvr+1];  /* y component */
                dc_dv[2] = diff * m_grad[3*mvr+2];  /* z component */
                bspline_update_grad_b (&bst->ssd, bxf, pidx, qidx, dc_dv);
        
                if (parms->debug) {
                    fprintf (val_fp, 
                        "%u %u %u %g %g %g\n", 
                        (unsigned int) fijk[0], 
                        (unsigned int) fijk[1], 
                        (unsigned int) fijk[2], 
                        f_img[fv], m_val, diff);
                    fprintf (dc_dv_fp, 
                        "%u %u %u %g %g %g %g\n", 
                        (unsigned int) fijk[0], 
                        (unsigned int) fijk[1], 
                        (unsigned int) fijk[2], 
                        diff, 
                        dc_dv[0], dc_dv[1], dc_dv[2]);
                }

                score_acc += diff * diff;
                ssd->num_vox++;
#endif
            } /* LOOP_THRU_ROI_X */
        } /* LOOP_THRU_ROI_Y */
    } /* LOOP_THRU_ROI_Z */

    if (parms->debug) {
        fclose (val_fp);
        fclose (dc_dv_fp);
        fclose (corr_fp);
    }

    /* Normalize score for MSE */
    bspline_score_normalize (bod, score_acc);

    ssd->time_smetric = timer->report ();
    delete timer;
}

#endif
