/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include "plm_config.h"
#include "rad_registration.h"
#include "xform.h"
#include "readmha.h"
#include "volume.h"
#include "demons.h"

void
do_gpuit_demons_stage_internal (Registration_Data* regd, 
				    Xform *xf_out, 
				    Xform *xf_in, 
				    Stage_Parms* stage)
{
    int d;
    DEMONS_Parms parms;
    Volume* fixed = regd->fixed_image->gpuit_float();
    Volume* moving = regd->moving_image->gpuit_float();
    Volume *moving_ss, *fixed_ss;
    Volume* moving_grad = 0;
    Volume* vf_out = 0;
    Volume* vf_in = 0;

    volume_convert_to_float (moving);		    /* Maybe not necessary? */
    volume_convert_to_float (fixed);		    /* Maybe not necessary? */

    printf ("SUBSAMPLE: %d %d %d\n", stage->resolution[0], stage->resolution[1], stage->resolution[2]);
    moving_ss = volume_subsample (moving, stage->resolution);
    fixed_ss = volume_subsample (fixed, stage->resolution);

    moving_grad = volume_make_gradient (moving_ss);

    demons_default_parms (&parms);
    parms.max_its = stage->max_its;
    parms.filter_std = stage->demons_std;
    parms.accel = stage->demons_acceleration;
    parms.homog = stage->demons_homogenization;
    for (d = 0; d < 3; d++) {
	parms.filter_width[d] = stage->demons_filter_width[d];
    }

    /* Transform input xform to gpuit vector field */
    if (xf_out->m_type == STAGE_TRANSFORM_NONE) {
	vf_in = 0;
    } else {
	xform_to_gpuit_vf (xf_out, xf_in, fixed_ss->dim, fixed_ss->offset, fixed_ss->pix_spacing);
	vf_in = xf_out->get_gpuit_vf();
    }

    /* Run demons */
    if (stage->impl_type == IMPLEMENTATION_GPUIT_CPU) {
	vf_out = demons (fixed_ss, moving_ss, moving_grad, vf_in, "CPU", &parms);
    } else {
	vf_out = demons (fixed_ss, moving_ss, moving_grad, vf_in, "GPU", &parms);
    }

    /* Do something with output vector field */
    xf_out->set_gpuit_vf (vf_out);

    volume_free (fixed_ss);
    volume_free (moving_ss);
    volume_free (moving_grad);
}

void
do_gpuit_demons_stage (Registration_Data* regd, 
			 Xform *xf_out, 
			 Xform *xf_in,
			 Stage_Parms* stage)
{
    do_gpuit_demons_stage_internal (regd, xf_out, xf_in, stage);
//    printf ("Deformation stats (out)\n");
//    deformation_stats (xf_out->get_itk_vf());
}
