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
#include "translation_optimize.h"
#include "translation_score.h"
#include "volume.h"
#include "volume_macros.h"
#include "volume_resample.h"
#include "xform.h"

static void
gridsearch_translation_old (
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
    lprintf ("Computing grid search extent.\n");
    float search_min[3];
    float search_max[3];
    for (int d = 0; d < 3; d++) {
        float mo = stage->gridsearch_min_overlap[d];
        if (mo < 0.1) { mo = 0.1; }
        else if (mo > 0.9) { mo = 0.9; }

        float mov_siz = moving->dim[d] * moving->spacing[d];
        float fix_siz = fixed->dim[d] * fixed->spacing[d];
        lprintf ("Dimension %d, mo=%g F=(%g, %g) M=(%g, %g)\n",
            d, mo, fixed->offset[d], fix_siz,
            moving->offset[d], mov_siz);
        
        if (fix_siz > mov_siz) {
            search_min[d] = moving->offset[d] - fixed->offset[d] 
                - fix_siz + mo * mov_siz;
            search_max[d] = moving->offset[d] - fixed->offset[d] 
                + mov_siz - mo * mov_siz;
        } else {
            search_min[d] = moving->offset[d] - fixed->offset[d] 
                - fix_siz + mo * fix_siz;
            search_max[d] = moving->offset[d] - fixed->offset[d] 
                + mov_siz - mo * fix_siz;
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

    /* Run grid search -- phase 1 */
    float best_translation[3] = { 0.f, 0.f, 0.f };
    float best_score = FLT_MAX;
    for (plm_long k = 0; k < num_steps[2]; k++) {
        for (plm_long j = 0; j < num_steps[1]; j++) {
            for (plm_long i = 0; i < num_steps[0]; i++) {
                float dxyz[3] = {
                    search_min[0] + i * search_step[0],
                    search_min[1] + j * search_step[1],
                    search_min[2] + k * search_step[2] };
                float score = translation_score (fixed, moving, dxyz);
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

    /* Run grid search -- phase 2 */
    for (int d = 0; d < 3; d++) {
        num_steps[d] = 4;
        search_step[d] = nominal_step / 2;
        search_min[d] = best_translation[d] - 1.5 * search_step[d];
    }
    for (plm_long k = 0; k < num_steps[2]; k++) {
        for (plm_long j = 0; j < num_steps[1]; j++) {
            for (plm_long i = 0; i < num_steps[0]; i++) {
                float dxyz[3] = {
                    search_min[0] + i * search_step[0],
                    search_min[1] + j * search_step[1],
                    search_min[2] + k * search_step[2] };
                float score = translation_score (fixed, moving, dxyz);
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

static void
gridsearch_translation (
    Xform::Pointer& xf_out, 
    Stage_parms* stage,
    const Volume::Pointer& fixed,
    const Volume::Pointer& moving)
{
    /* GCS FIX: region of interest is not used */

    /* GCS FIX: This algorithm will not work with tilted images.
       For these cases, we need to use bounding box to compute 
       search extent. */

    /* Compute maximum search extent */
    lprintf ("Computing grid search extent.\n");
    float search_min[3];
    float search_max[3];
    for (int d = 0; d < 3; d++) {
        float mo = stage->gridsearch_min_overlap[d];
        if (mo < 0.1) { mo = 0.1; }
        else if (mo > 0.9) { mo = 0.9; }

        float mov_siz = moving->dim[d] * moving->spacing[d];
        float fix_siz = fixed->dim[d] * fixed->spacing[d];
        lprintf ("Dimension %d, mo=%g F=(%g, %g) M=(%g, %g)\n",
            d, mo, fixed->offset[d], fix_siz,
            moving->offset[d], mov_siz);
        
        if (fix_siz > mov_siz) {
            search_min[d] = moving->offset[d] - fixed->offset[d] 
                - fix_siz + mo * mov_siz;
            search_max[d] = moving->offset[d] - fixed->offset[d] 
                + mov_siz - mo * mov_siz;
        } else {
            search_min[d] = moving->offset[d] - fixed->offset[d] 
                - fix_siz + mo * fix_siz;
            search_max[d] = moving->offset[d] - fixed->offset[d] 
                + mov_siz - mo * fix_siz;
        }
    }
    lprintf ("Native grid search extent: "
        "(%g, %g), (%g, %g), (%g, %g)\n",
        search_min[0], search_max[0], 
        search_min[1], search_max[1], 
        search_min[2], search_max[2]);

    /* Get default value */
    TranslationTransformType::Pointer old_trn = xf_out->get_trn ();
    float best_translation[3] = { 0.f, 0.f, 0.f };
    best_translation[0] = old_trn->GetParameters()[0];
    best_translation[1] = old_trn->GetParameters()[1];
    best_translation[2] = old_trn->GetParameters()[2];
    float best_score = translation_score (fixed, moving, best_translation);
    lprintf ("[%g %g %g] %g *\n", 
        best_translation[0], best_translation[1], best_translation[2], 
        best_score);

    /* Compute search locations */
    int num_steps[3] = { 0, 0, 0 };
    float search_step[3] = { 0.f, 0.f, 0.f };
    float max_range = 0.f;
    for (int d = 0; d < 3; d++) {
        float search_range = search_max[d] - search_min[d];
        if (search_range > max_range) {
            max_range = search_range;
        }
    }
    float nominal_step = max_range / 5;

    if (stage->gridsearch_strategy == GRIDSEARCH_STRATEGY_GLOBAL) {
        for (int d = 0; d < 3; d++) {
            float search_range = search_max[d] - search_min[d];
            num_steps[d] = ROUND_INT (search_range / nominal_step) + 1;
            if (num_steps[d] > 1) {
                search_step[d] = search_range / (num_steps[d] - 1);
            }
        }
    }
    else {
        for (int d = 0; d < 3; d++) {
            num_steps[d] = 4;
            if (stage->gridsearch_step_size_type == GRIDSEARCH_STEP_SIZE_AUTO)
            {
                search_step[d] = 0.6 * nominal_step;
            } else {
                search_step[d] = stage->gridsearch_step_size[d];
            }
            search_min[d] = best_translation[d] - 1.5 * search_step[d];
        }
    }

    /* Run grid search */
    for (plm_long k = 0; k < num_steps[2]; k++) {
        for (plm_long j = 0; j < num_steps[1]; j++) {
            for (plm_long i = 0; i < num_steps[0]; i++) {
                float translation[3] = {
                    search_min[0] + i * search_step[0],
                    search_min[1] + j * search_step[1],
                    search_min[2] + k * search_step[2] };
                float score = translation_score (fixed, moving, translation);
                lprintf ("[%g %g %g] %g", 
                    translation[0], translation[1], translation[2], score);
                if (score < best_score) {
                    best_score = score;
                    best_translation[0] = translation[0];
                    best_translation[1] = translation[1];
                    best_translation[2] = translation[2];
                    lprintf (" *");
                }
                lprintf ("\n");
            }
        }
    }

    /* Find the best translation */
    TranslationTransformType::ParametersType xfp(3);
    xfp[0] = best_translation[0];
    xfp[1] = best_translation[1];
    xfp[2] = best_translation[2];

    /* Fixate translation into xform */
    TranslationTransformType::Pointer new_trn = TranslationTransformType::New();
    new_trn->SetParameters(xfp);
    xf_out->set_trn (new_trn);
}

Xform::Pointer
translation_stage (
    Registration_data* regd, 
    const Xform::Pointer& xf_in,
    Stage_parms* stage)
{
    Xform::Pointer xf_out = Xform::New ();
    Plm_image_header pih;

    Volume::Pointer& fixed = regd->fixed_image->get_volume_float ();
    Volume::Pointer& moving = regd->moving_image->get_volume_float ();
    Volume::Pointer moving_ss;
    Volume::Pointer fixed_ss;

    fixed->convert (PT_FLOAT);              /* Maybe not necessary? */
    moving->convert (PT_FLOAT);             /* Maybe not necessary? */

    lprintf ("SUBSAMPLE: (%g %g %g), (%g %g %g)\n", 
	stage->resample_rate_fixed[0], stage->resample_rate_fixed[1], 
	stage->resample_rate_fixed[2], stage->resample_rate_moving[0], 
	stage->resample_rate_moving[1], stage->resample_rate_moving[2]
    );
    moving_ss = volume_subsample_vox_legacy (
        moving, stage->resample_rate_moving);
    fixed_ss = volume_subsample_vox_legacy (
        fixed, stage->resample_rate_fixed);

    /* Transform input xform to itk translation */
    xform_to_trn (xf_out.get(), xf_in.get(), &pih);

    /* Run the translation optimizer */
    gridsearch_translation (xf_out, stage, fixed_ss, moving_ss);

    return xf_out;
}
