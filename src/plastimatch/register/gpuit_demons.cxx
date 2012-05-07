/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#include "plmbase.h"

#include "demons.h"
#include "plmregister_config.h"
#include "plm_stages.h"
#include "plm_parms.h"
#include "registration_data.h"

void
do_gpuit_demons_stage_internal (
    Registration_data* regd, 
    Xform *xf_out, 
    Xform *xf_in, 
    Stage_parms* stage)
{
    int d;
    Demons_parms parms;
    Plm_image_header pih;

    Volume* fixed = regd->fixed_image->gpuit_float();
    Volume* moving = regd->moving_image->gpuit_float();
    Volume *moving_ss, *fixed_ss;
    Volume* moving_grad = 0;
    Volume* vf_out = 0;
    Volume* vf_in = 0;

    volume_convert_to_float (moving);		    /* Maybe not necessary? */
    volume_convert_to_float (fixed);		    /* Maybe not necessary? */

    printf ("SUBSAMPLE: (%d %d %d), (%d %d %d)\n", 
	stage->fixed_subsample_rate[0], stage->fixed_subsample_rate[1], 
	stage->fixed_subsample_rate[2], stage->moving_subsample_rate[0], 
	stage->moving_subsample_rate[1], stage->moving_subsample_rate[2]
    );
    moving_ss = volume_subsample (moving, stage->moving_subsample_rate);
    fixed_ss = volume_subsample (fixed, stage->fixed_subsample_rate);

    moving_grad = volume_make_gradient (moving_ss);

    demons_default_parms (&parms);
    parms.max_its = stage->max_its;
    parms.filter_std = stage->demons_std;
    parms.accel = stage->demons_acceleration;
    parms.homog = stage->demons_homogenization;
    parms.threading = stage->threading_type;
    for (d = 0; d < 3; d++) {
	parms.filter_width[d] = stage->demons_filter_width[d];
    }

    /* Transform input xform to gpuit vector field */
    if (xf_in->m_type == STAGE_TRANSFORM_NONE) {
	vf_in = 0;
    } else {
        pih.set_from_gpuit (fixed_ss->dim, 
            fixed_ss->offset, fixed_ss->spacing, 
            fixed_ss->direction_cosines);
	xform_to_gpuit_vf (xf_out, xf_in, &pih);
	vf_in = xf_out->get_gpuit_vf();
    }

    /* Run demons */
    vf_out = demons (fixed_ss, moving_ss, moving_grad, vf_in, &parms);

    /* Do something with output vector field */
    xf_out->set_gpuit_vf (vf_out);

    delete fixed_ss;
    delete moving_ss;
    delete moving_grad;
}

void
do_gpuit_demons_stage (
    Registration_data* regd, 
    Xform *xf_out, 
    Xform *xf_in,
    Stage_parms* stage)
{
    do_gpuit_demons_stage_internal (regd, xf_out, xf_in, stage);
    //    printf ("Deformation stats (out)\n");
    //    deformation_stats (xf_out->get_itk_vf());
}
