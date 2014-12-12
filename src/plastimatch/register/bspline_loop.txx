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

/* Similar to above, but more complex */
template< class Bspline_loop_class >
void
bspline_loop_k (
    Bspline_loop_class& bspline_loop_class,
    Bspline_optimize *bod
)
{
    Bspline_parms *parms = bod->get_bspline_parms ();
    Bspline_state *bst = bod->get_bspline_state ();
    Bspline_xform *bxf = bod->get_bspline_xform ();

    Volume *fixed = parms->fixed;
    Volume *moving = parms->moving;
    Volume* fixed_roi  = parms->fixed_roi;
    Volume* moving_roi = parms->moving_roi;

    Bspline_score* ssd = &bst->ssd;
    plm_long fijk[3], fidx;     /* Indices within fixed image (vox) */
    float mijk[3];              /* Indices within moving image (vox) */
    float fxyz[3];              /* Position within fixed image (mm) */
    float mxyz[3];              /* Position within moving image (mm) */
    plm_long mijk_f[3], mvf;    /* Floor */
    plm_long mijk_r[3];         /* Round */
    plm_long p[3], pidx;        /* Region index of fixed voxel */
    plm_long q[3], qidx;        /* Offset index of fixed voxel */

    float li_1[3];           /* Fraction of interpolant in lower index */
    float li_2[3];           /* Fraction of interpolant in upper index */
    float* f_img = (float*) fixed->img;
    float* m_img = (float*) moving->img;
    float dxyz[3];

    FILE* val_fp = 0;
    FILE* dc_dv_fp = 0;
    FILE* corr_fp = 0;

    if (parms->debug) {
        std::string fn;

        fn = string_format ("%s/%02d_%03d_%03d_dc_dv.csv",
            parms->debug_dir.c_str(), parms->debug_stage, bst->it, 
            bst->feval);
        dc_dv_fp = plm_fopen (fn.c_str(), "wb");

        fn = string_format ("%s/%02d_%03d_%03d_val.csv",
            parms->debug_dir.c_str(), parms->debug_stage, bst->it, 
            bst->feval);
        val_fp = plm_fopen (fn.c_str(), "wb");

        fn = string_format ("%s/%02d_%03d_%03d_corr.csv",
            parms->debug_dir.c_str(), parms->debug_stage, bst->it, 
            bst->feval);
        corr_fp = plm_fopen (fn.c_str(), "wb");
    }
    
    LOOP_Z (fijk, fxyz, fixed) {
        p[2] = REGION_INDEX_Z (fijk, bxf);
        q[2] = REGION_OFFSET_Z (fijk, bxf);
        LOOP_Y (fijk, fxyz, fixed) {
            p[1] = REGION_INDEX_Y (fijk, bxf);
            q[1] = REGION_OFFSET_Y (fijk, bxf);
            LOOP_X (fijk, fxyz, fixed) {
                p[0] = REGION_INDEX_X (fijk, bxf);
                q[0] = REGION_OFFSET_X (fijk, bxf);

                /* Discard fixed image voxels outside of roi */
                if (fixed_roi) {
                    if (!inside_roi (fxyz, fixed_roi)) continue;
                }

                /* Get B-spline deformation vector */
                pidx = volume_index (bxf->rdims, p);
                qidx = volume_index (bxf->vox_per_rgn, q);
                bspline_interp_pix_b (dxyz, bxf, pidx, qidx);

                /* Find correspondence in moving image */
                int rc;
                rc = bspline_find_correspondence_dcos_roi (
                    mxyz, mijk, fxyz, dxyz, moving, moving_roi);

                /* If voxel is not inside moving image */
                if (!rc) continue;

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

                /* Compute interpolation fractions */
                li_clamp_3d (mijk, mijk_f, mijk_r, li_1, li_2, moving);

                /* Compute linear index of fixed image voxel */
                fidx = volume_index (fixed->dim, fijk);

                /* Find linear index of "corner voxel" in moving image */
                mvf = volume_index (moving->dim, mijk_f);

                /* Run the target function */
                bspline_loop_class.loop_function (
                    bod, bxf, ssd, 
                    moving, f_img, m_img, 
                    //m_grad, 
                    fidx, mvf, mijk_r, 
                    pidx, qidx, li_1, li_2);

#if defined (commentout)
                /* MSE */
                /* Compute moving image intensity using linear interpolation */
                /* Macro is slightly faster than function */
                LI_VALUE (m_val, 
                    li_1[0], li_2[0],
                    li_1[1], li_2[1],
                    li_1[2], li_2[2],
                    mvf, m_img, moving);

                /* This replaces the commented out code */
                double score_incr;
                mvr = volume_index (moving->dim, mijk_r);
                bspline_score_function (
                    score_incr,
                    dc_dv,
                    m_val,
                    f_img[fv],
                    &m_grad[3*mvr]);
                bspline_update_grad_b (&bst->ssd, bxf, pidx, qidx, dc_dv);
                score_acc += score_incr;
                ssd->num_vox++;

                /* MI PASS #1 */
                LI_VALUE (m_val, 
                    li_1[0], li_2[0],
                    li_1[1], li_2[1],
                    li_1[2], li_2[2],
                    mvf, m_img, moving
                );
                /* PARTIAL VALUE INTERPOLATION - 8 neighborhood */
                bspline_mi_hist_add_pvi_8 (
                    mi_hist, fixed, moving, 
                    fv, mvf, li_1, li_2
                );

                
                /* MI PASS #2 */
                /* Compute dc_dv */
                bspline_mi_pvi_8_dc_dv_dcos (
                    dc_dv, mi_hist, bst,
                    fixed, moving, 
                    fv, mvf, mijk,
                    num_vox_f, li_1, li_2
                );
                bspline_update_grad_b (&bst->ssd, bxf, pidx, qidx, dc_dv);
#endif
                
            } /* LOOP_THRU_ROI_X */
        } /* LOOP_THRU_ROI_Y */
    } /* LOOP_THRU_ROI_Z */

    if (parms->debug) {
        fclose (val_fp);
        fclose (dc_dv_fp);
        fclose (corr_fp);
    }

}

#endif
