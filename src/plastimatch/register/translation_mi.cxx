/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plmregister_config.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "interpolate.h"
#include "interpolate_macros.h"
#include "logfile.h"
#include "plm_image.h"
#include "plm_image_header.h"
#include "registration_data.h"
#include "stage_parms.h"
#include "translation_mi.h"
#include "volume.h"
#include "volume_macros.h"
#include "volume_resample.h"
#include "xform.h"

float
translation_mi (
    const Stage_parms *stage,
    const Volume::Pointer& fixed,
    const Volume::Pointer& moving,
    const float dxyz[3])
{
#if defined (commentout)
    plm_long fijk[3], fv;         /* Indices within fixed image (vox) */
    float fxyz[3];                /* Position within fixed image (mm) */
    float mijk[3];                /* Indices within moving image (vox) */
    float mxyz[3];                /* Position within moving image (mm) */

    float li_1[3];                /* Fraction of interpolant in lower index */
    float li_2[3];                /* Fraction of interpolant in upper index */
    plm_long mijk_f[3], mvf;      /* Floor */
    plm_long mijk_r[3];           /* Round */
    float m_val;

    Volume *mvol = moving.get();
    float* f_img = (float*) fixed->img;
    float* m_img = (float*) moving->img;

    double score_acc = 0.0;
    plm_long num_vox = 0;

    LOOP_Z (fijk, fxyz, fixed) {
        LOOP_Y (fijk, fxyz, fixed) {
            LOOP_X (fijk, fxyz, fixed) {

                /* Compute moving image coordinate of fixed image voxel */
                mxyz[2] = fxyz[2] + dxyz[2] - moving->offset[2];
                mxyz[1] = fxyz[1] + dxyz[1] - moving->offset[1];
                mxyz[0] = fxyz[0] + dxyz[0] - moving->offset[0];
                mijk[2] = PROJECT_Z (mxyz, moving->proj);
                mijk[1] = PROJECT_Y (mxyz, moving->proj);
                mijk[0] = PROJECT_X (mxyz, moving->proj);

                if (mijk[2] < -0.5 || mijk[2] > moving->dim[2] - 0.5) continue;
                if (mijk[1] < -0.5 || mijk[1] > moving->dim[1] - 0.5) continue;
                if (mijk[0] < -0.5 || mijk[0] > moving->dim[0] - 0.5) continue;

                /* Compute interpolation fractions */
                li_clamp_3d (mijk, mijk_f, mijk_r, li_1, li_2, mvol);

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

                /* Compute intensity difference */
                float diff = m_val - f_img[fv];

                score_acc += diff * diff;
                num_vox++;
            }
        }
    }

    /* Normalize score */
    const int MIN_VOX = 1;
    float final_metric;
    if (num_vox < MIN_VOX) {
        final_metric = FLT_MAX;
    } else {
        final_metric = score_acc / num_vox;
    }

    return final_metric;
#endif

    Bspline_mi_hist_set mi_hist (
        stage->mi_hist_type,
        stage->mi_hist_fixed_bins,
        stage->mi_hist_moving_bins);

    return 0.f;
        
#if defined (commentout)
    Bspline_parms *parms = bod->get_bspline_parms ();
    Bspline_state *bst = bod->get_bspline_state ();
    Bspline_xform *bxf = bod->get_bspline_xform ();

    Volume *fixed = parms->fixed;
    Volume *moving = parms->moving;

    Bspline_score* ssd = &bst->ssd;
    Bspline_mi_hist_set* mi_hist = bst->mi_hist;
    long pidx;
    float num_vox_f;

    float mse_score = 0.0f;
    double* f_hist = mi_hist->f_hist;
    double* m_hist = mi_hist->m_hist;
    double* j_hist = mi_hist->j_hist;
    double mhis = 0.0f;      /* Moving histogram incomplete sum */
    double jhis = 0.0f;      /* Joint  histogram incomplete sum */

    int num_threads;
    double* f_hist_omp = NULL;
    double* m_hist_omp = NULL;
    double* j_hist_omp = NULL;

    plm_long cond_size = 64*bxf->num_knots*sizeof(float);
    float* cond_x = (float*)malloc(cond_size);
    float* cond_y = (float*)malloc(cond_size);
    float* cond_z = (float*)malloc(cond_size);

    Plm_timer* timer = new Plm_timer;
    timer->start ();

    memset (f_hist, 0, mi_hist->fixed.bins * sizeof(double));
    memset (m_hist, 0, mi_hist->moving.bins * sizeof(double));
    memset (j_hist, 0, mi_hist->fixed.bins * mi_hist->moving.bins * sizeof(double));
    memset(cond_x, 0, cond_size);
    memset(cond_y, 0, cond_size);
    memset(cond_z, 0, cond_size);

#pragma omp parallel
#pragma omp master
    {
        num_threads = omp_get_num_threads ();
        f_hist_omp = (double*) malloc (num_threads * sizeof (double) * mi_hist->fixed.bins);
        m_hist_omp = (double*) malloc (num_threads * sizeof (double) * mi_hist->moving.bins);
        j_hist_omp = (double*) malloc (num_threads * sizeof (double) * mi_hist->fixed.bins * mi_hist->moving.bins);
        memset (f_hist_omp, 0, num_threads * sizeof (double) * mi_hist->fixed.bins);
        memset (m_hist_omp, 0, num_threads * sizeof (double) * mi_hist->moving.bins);
        memset (j_hist_omp, 0, num_threads * sizeof (double) * mi_hist->fixed.bins * mi_hist->moving.bins);
    }

    /* PASS 1 - Accumulate histogram */
#pragma omp parallel for
    LOOP_THRU_VOL_TILES (pidx, bxf) {
        int rc;
        plm_long fijk[3], fidx;
        float mijk[3];
        float fxyz[3];
        float mxyz[3];
        plm_long mijk_f[3], midx_f;      /* Floor */
        plm_long mijk_r[3];           /* Round */
        plm_long p[3];
        plm_long q[3];
        float dxyz[3];
        float li_1[3];           /* Fraction of interpolant in lower index */
        float li_2[3];           /* Fraction of interpolant in upper index */

        int thread_num = omp_get_thread_num ();

        /* Get tile indices from linear index */
        COORDS_FROM_INDEX (p, pidx, bxf->rdims);

        /* Serial through the voxels in a tile */
        LOOP_THRU_TILE_Z (q, bxf) {
            LOOP_THRU_TILE_Y (q, bxf) {
                LOOP_THRU_TILE_X (q, bxf) {
                    
                    /* Construct coordinates into fixed image volume */
                    GET_VOL_COORDS (fijk, p, q, bxf);
                    
                    /* Check to make sure the indices are valid (inside volume) */
                    if (fijk[0] >= bxf->roi_offset[0] + bxf->roi_dim[0]) { continue; }
                    if (fijk[1] >= bxf->roi_offset[1] + bxf->roi_dim[1]) { continue; }
                    if (fijk[2] >= bxf->roi_offset[2] + bxf->roi_dim[2]) { continue; }

                    /* Compute space coordinates of fixed image voxel */
                    GET_REAL_SPACE_COORDS (fxyz, fijk, bxf);

                    /* Compute deformation vector (dxyz) for voxel */
                    bspline_interp_pix_c (dxyz, bxf, pidx, q);

                    /* Find correspondence in moving image */
                    rc = bspline_find_correspondence (mxyz, mijk, fxyz, dxyz, moving);

                    /* If voxel is not inside moving image */
                    if (!rc) continue;

                    /* Get tri-linear interpolation fractions */
                    li_clamp_3d (mijk, mijk_f, mijk_r, li_1, li_2, moving);
                    
                    /* Constrcut the fixed image linear index within volume space */
                    fidx = volume_index (fixed->dim, fijk);

                    /* Find linear index the corner voxel used to identifiy the
                     * neighborhood of the moving image voxels corresponding
                     * to the current fixed image voxel */
                    midx_f = volume_index (moving->dim, mijk_f);

                    /* Add to histogram */

                    bspline_mi_hist_add_pvi_8_omp_v2 (
                        mi_hist, f_hist_omp, m_hist_omp, j_hist_omp,
                        fixed, moving, fidx, midx_f, li_1, li_2, thread_num);
                }
            }
        }   // tile
    }   // openmp

    /* Merge the OpenMP histogram copies */
    for (plm_long b=0; b<mi_hist->fixed.bins; b++) {
        for (int c=0; c<num_threads; c++) {
            f_hist[b] += f_hist_omp[c*mi_hist->fixed.bins + b];
        }
    }
    for (plm_long b=0; b<mi_hist->moving.bins; b++) {
        for (int c=0; c<num_threads; c++) {
            m_hist[b] += m_hist_omp[c*mi_hist->moving.bins + b];
        }
    }
    for (plm_long j=0; j<mi_hist->fixed.bins; j++) {
        for (plm_long i=0; i<mi_hist->moving.bins; i++) {
            for (int c=0; c<num_threads; c++) {
                j_hist[j*mi_hist->moving.bins+i] += j_hist_omp[c*mi_hist->moving.bins*mi_hist->fixed.bins + j*mi_hist->moving.bins + i];
            }
        }
    }

    /* Compute num_vox and find fullest fixed hist bin */
    for (plm_long i=0; i<mi_hist->fixed.bins; i++) {
        if (f_hist[i] > f_hist[mi_hist->fixed.big_bin]) {
            mi_hist->fixed.big_bin = i;
        }
        ssd->num_vox += f_hist[i];
    }

    /* Fill in the missing histogram bin */
    for (plm_long i=0; i<mi_hist->moving.bins; i++) {
        mhis += m_hist[i];
    }
    m_hist[mi_hist->moving.big_bin] = (double)ssd->num_vox - mhis;


    /* Look for the biggest moving histogram bin */
//    printf ("moving.big_bin [%i -> ", mi_hist->moving.big_bin);
    for (plm_long i=0; i<mi_hist->moving.bins; i++) {
        if (m_hist[i] > m_hist[mi_hist->moving.big_bin]) {
            mi_hist->moving.big_bin = i;
        }
    }
//    printf ("%i]\n", mi_hist->moving.big_bin);


    /* Fill in the missing jnt hist bin */
    for (plm_long j=0; j<mi_hist->fixed.bins; j++) {
        for (plm_long i=0; i<mi_hist->moving.bins; i++) {
            jhis += j_hist[j*mi_hist->moving.bins + i];
        }
    }
    j_hist[mi_hist->joint.big_bin] = (double)ssd->num_vox - jhis;

    
    /* Look for the biggest joint histogram bin */
//    printf ("joint.big_bin [%i -> ", mi_hist->joint.big_bin);
    for (plm_long j=0; j<mi_hist->fixed.bins; j++) {
        for (plm_long i=0; i<mi_hist->moving.bins; i++) {
            if (j_hist[j*mi_hist->moving.bins + i] > j_hist[mi_hist->joint.big_bin]) {
                mi_hist->joint.big_bin = j*mi_hist->moving.bins + i;
            }
        }
    }
//    printf ("%i]\n", mi_hist->joint.big_bin);
    


    /* Draw histogram images if user wants them */
    if (parms->xpm_hist_dump) {
        dump_xpm_hist (mi_hist, parms->xpm_hist_dump, bst->it);
    }

    /* Display histrogram stats in debug mode */
    if (parms->debug) {
        double tmp;
        long zz;
        for (zz=0,tmp=0; zz < mi_hist->fixed.bins; zz++) {
            tmp += f_hist[zz];
        }
        printf ("f_hist total: %f\n", tmp);

        for (zz=0,tmp=0; zz < mi_hist->moving.bins; zz++) {
            tmp += m_hist[zz];
        }
        printf ("m_hist total: %f\n", tmp);

        for (zz=0,tmp=0; zz < mi_hist->moving.bins * mi_hist->fixed.bins; zz++) {
            tmp += j_hist[zz];
        }
        printf ("j_hist total: %f\n", tmp);
    }

    /* Compute score */
    ssd->smetric = mi_hist_score_omp (mi_hist, ssd->num_vox);
    num_vox_f = (float) ssd->num_vox;

    /* PASS 2 - Compute Gradient (Parallel across tiles) */
#pragma omp parallel for
    LOOP_THRU_VOL_TILES (pidx, bxf) {
        int rc;
        plm_long fijk[3], fidx;
        float mijk[3];
        float fxyz[3];
        float mxyz[3];
        plm_long mijk_f[3], midx_f;      /* Floor */
        plm_long mijk_r[3];           /* Round */
        plm_long p[3];
        plm_long q[3];
        float dxyz[3];
        float li_1[3];           /* Fraction of interpolant in lower index */
        float li_2[3];           /* Fraction of interpolant in upper index */
        float dc_dv[3];
        float sets_x[64];
        float sets_y[64];
        float sets_z[64];

        memset(sets_x, 0, 64*sizeof(float));
        memset(sets_y, 0, 64*sizeof(float));
        memset(sets_z, 0, 64*sizeof(float));

        /* Get tile indices from linear index */
        COORDS_FROM_INDEX (p, pidx, bxf->rdims);

        /* Serial through the voxels in a tile */
        LOOP_THRU_TILE_Z (q, bxf) {
            LOOP_THRU_TILE_Y (q, bxf) {
                LOOP_THRU_TILE_X (q, bxf) {
                    
                    /* Construct coordinates into fixed image volume */
                    GET_VOL_COORDS (fijk, p, q, bxf);
                    
                    /* Check to make sure the indices are valid (inside volume) */
                    if (fijk[0] >= bxf->roi_offset[0] + bxf->roi_dim[0]) { continue; }
                    if (fijk[1] >= bxf->roi_offset[1] + bxf->roi_dim[1]) { continue; }
                    if (fijk[2] >= bxf->roi_offset[2] + bxf->roi_dim[2]) { continue; }

                    /* Compute space coordinates of fixed image voxel */
                    GET_REAL_SPACE_COORDS (fxyz, fijk, bxf);

                    /* Compute deformation vector (dxyz) for voxel */
                    bspline_interp_pix_c (dxyz, bxf, pidx, q);

                    /* Find correspondence in moving image */
                    rc = bspline_find_correspondence (mxyz, mijk, fxyz, dxyz, moving);

                    /* If voxel is not inside moving image */
                    if (!rc) continue;

                    /* Get tri-linear interpolation fractions */
                    li_clamp_3d (mijk, mijk_f, mijk_r, li_1, li_2, moving);
                    
                    /* Constrcut the fixed image linear index within volume space */
                    fidx = volume_index (fixed->dim, fijk);

                    /* Find linear index the corner voxel used to identifiy the
                     * neighborhood of the moving image voxels corresponding
                     * to the current fixed image voxel */
                    midx_f = volume_index (moving->dim, mijk_f);

                    /* Compute dc_dv */
                    bspline_mi_pvi_8_dc_dv (dc_dv, mi_hist, bst, fixed, moving, 
                        fidx, midx_f, mijk, num_vox_f, li_1, li_2);

                    /* Update condensed tile sets */
                    bspline_update_sets_b (sets_x, sets_y, sets_z,
                        q, dc_dv, bxf);
                }
            }
        }   // tile

        /* We now have a tile of condensed dc_dv values (64 values).
         * Let's put each one in the proper slot within the control
         * point bin its belogs to */
        bspline_sort_sets (cond_x, cond_y, cond_z,
            sets_x, sets_y, sets_z,
            pidx, bxf);
    }   // openmp

    /* Now we have a ton of bins and each bin's 64 slots are full.
     * Let's sum each bin's 64 slots.  The result with be dc_dp. */
    bspline_condense_grad (cond_x, cond_y, cond_z,
        bxf, ssd);

    free (cond_x);
    free (cond_y);
    free (cond_z);

    mse_score = mse_score / ssd->num_vox;
    
    ssd->time_smetric = timer->report ();
    delete timer;
#endif
}
