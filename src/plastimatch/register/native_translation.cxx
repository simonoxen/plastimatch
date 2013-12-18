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
#include "volume.h"
#include "volume_macros.h"
#include "volume_resample.h"
#include "xform.h"

static float
native_translation_score (
    const Volume::Pointer& fixed,
    const Volume::Pointer& moving,
    const float dxyz[3])
{
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
}

static void
native_translation (
    Xform *xf_out, 
    Stage_parms* stage,
    const Volume::Pointer& fixed,
    const Volume::Pointer& moving)
{
    TranslationTransformType::Pointer trn = TranslationTransformType::New();
    TranslationTransformType::ParametersType xfp(3);

    /* GCS FIX: region of interest is not used */

    /* GCS FIX: This algorithm will not work with tilted images.
       For these cases, we need to use bounding box to compute 
       search extent. */

    /* Compute search extent -- range of search is up to 50% overlap 
       in any one dimension */
    lprintf ("min_overlap = %g %g %g\n",
        stage->gridsearch_min_overlap[0],
        stage->gridsearch_min_overlap[1],
        stage->gridsearch_min_overlap[2]);

    float search_min[3];
    float search_max[3];
    for (int d = 0; d < 3; d++) {
        float mo = stage->gridsearch_min_overlap[d];
        float mov_siz = moving->dim[d] *  moving->spacing[d];
        float fix_siz = fixed->dim[d] *  fixed->spacing[d];
        if (fix_siz > mov_siz) {
            search_min[d] = fixed->offset[d] - moving->offset[d] 
                - mov_siz + mo * mov_siz;
            search_max[d] = fixed->offset[d] - moving->offset[d] 
                + fix_siz - mo * mov_siz;
        } else {
            search_min[d] = fixed->offset[d] - moving->offset[d] 
                - mov_siz + mo * fix_siz;
            search_max[d] = fixed->offset[d] - moving->offset[d] 
                + fix_siz - mo * fix_siz;
        }
    }
    lprintf ("Native grid search extent: "
        "(%g, %g), (%g, %g), (%g, %g)\n",
        search_min[0], search_max[0], 
        search_min[1], search_max[1], 
        search_min[2], search_max[2]);

    /* Compute search intervals */
    float max_range = 0.f;
    for (int d = 0; d < 3; d++) {
        float search_range = search_max[d] - search_min[d];
        if (search_range > max_range) {
            max_range = search_range;
        }
    }
    int num_steps[3];
    float search_step[3] = { 0.f, 0.f, 0.f };
    float nominal_step = max_range / 5;
    for (int d = 0; d < 3; d++) {
        float search_range = search_max[d] - search_min[d];
        num_steps[d] = ROUND_INT (search_range / nominal_step) + 1;
        if (num_steps[d] > 1) {
            search_step[d] = search_range / (num_steps[d] - 1);
        }
    }

    float best_translation[3] = { 0.f, 0.f, 0.f };
    float best_score = FLT_MAX;
    for (plm_long k = 0; k < num_steps[2]; k++) {
        for (plm_long j = 0; j < num_steps[1]; j++) {
            for (plm_long i = 0; i < num_steps[0]; i++) {
                float dxyz[3] = {
                    search_min[0] + i * search_step[0],
                    search_min[1] + j * search_step[1],
                    search_min[2] + k * search_step[2] };
                float score = native_translation_score (fixed, moving, dxyz);
                lprintf ("[%g %g %g] %g", 
                    dxyz[0], dxyz[1], dxyz[2], score);
                if (score < best_score) {
                    best_score = score;
                    best_translation[0] = dxyz[0];
                    best_translation[1] = dxyz[1];
                    best_translation[2] = dxyz[2];
                    lprintf (" *");
                }
                lprintf ("\n");
            }
        }
    }

    /* Find the best translation */
    xfp[0] = best_translation[0];
    xfp[1] = best_translation[1];
    xfp[2] = best_translation[2];

    /* Fixate translation into xform */
    trn->SetParameters(xfp);
    xf_out->set_trn (trn);
}

void
native_translation_stage (
    Registration_data* regd, 
    Xform *xf_out, 
    Xform *xf_in, 
    Stage_parms* stage)
{
    Plm_image_header pih;

    Volume* fixed = regd->fixed_image->get_vol_float ();
    Volume* moving = regd->moving_image->get_vol_float ();
    Volume::Pointer moving_ss;
    Volume::Pointer fixed_ss;

    volume_convert_to_float (moving);		    /* Maybe not necessary? */
    volume_convert_to_float (fixed);		    /* Maybe not necessary? */

    lprintf ("SUBSAMPLE: (%g %g %g), (%g %g %g)\n", 
	stage->fixed_subsample_rate[0], stage->fixed_subsample_rate[1], 
	stage->fixed_subsample_rate[2], stage->moving_subsample_rate[0], 
	stage->moving_subsample_rate[1], stage->moving_subsample_rate[2]
    );
    moving_ss = volume_subsample_vox_legacy (
        moving, stage->moving_subsample_rate);
    fixed_ss = volume_subsample_vox_legacy (
        fixed, stage->fixed_subsample_rate);

    /* Transform input xform to gpuit vector field */
    if (xf_in->m_type == STAGE_TRANSFORM_NONE) {
        /* Do nothing */
    } else {
        /* Do something, tbd */
    }

    /* Run the translation optimizer */
    native_translation (xf_out, stage, fixed_ss, moving_ss);
}
