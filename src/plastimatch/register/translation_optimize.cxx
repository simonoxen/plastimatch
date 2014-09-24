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
gridsearch_translation (
    Xform::Pointer& xf_out, 
    const Stage_parms* stage,
    Stage_parms* auto_parms,
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

    /* Compute search range */
    int num_steps[3] = { 0, 0, 0 };
    float search_step[3] = { 0.f, 0.f, 0.f };
    float max_range = 0.f;
    for (int d = 0; d < 3; d++) {
        float search_range = search_max[d] - search_min[d];
        if (search_range > max_range) {
            max_range = search_range;
        }
    }

    /* Identify search strategy, and compute step size */
    Gridsearch_strategy_type strategy = stage->gridsearch_strategy;
    if (strategy == GRIDSEARCH_STRATEGY_AUTO) {
        strategy = auto_parms->gridsearch_strategy;
    }
    if (strategy == GRIDSEARCH_STRATEGY_GLOBAL || 
        strategy == GRIDSEARCH_STRATEGY_AUTO)
    {
        lprintf ("Global grid search\n");
        float nominal_step = max_range / 5;
        for (int d = 0; d < 3; d++) {
            float search_range = search_max[d] - search_min[d];
            num_steps[d] = ROUND_INT (search_range / nominal_step) + 1;
            if (num_steps[d] > 1) {
                search_step[d] = search_range / (num_steps[d] - 1);
            }
        }
    }
    else {
        lprintf ("Local grid search\n");
        for (int d = 0; d < 3; d++) {
            num_steps[d] = 4;
            if (stage->gridsearch_step_size_type == GRIDSEARCH_STEP_SIZE_AUTO)
            {
                search_step[d] = auto_parms->gridsearch_step_size[d];
            } else {
                search_step[d] = stage->gridsearch_step_size[d];
            }
            search_min[d] = best_translation[d] - 1.5 * search_step[d];
        }
    }

    /* Update auto parms */
    auto_parms->gridsearch_strategy = GRIDSEARCH_STRATEGY_LOCAL;
    for (int d = 0; d < 3; d++) {
        auto_parms->gridsearch_step_size[d] = 0.6 * search_step[d];
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
    const Stage_parms* stage)
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
    gridsearch_translation (xf_out, stage, 
        regd->get_auto_parms (),
        fixed_ss, moving_ss);

    return xf_out;
}
