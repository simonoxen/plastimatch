/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plm_config.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "bspline_opts.h"
#include "bspline.h"
#if defined (HAVE_F2C_LIBRARY)
#include "bspline_optimize_lbfgsb.h"
#endif
#include "readmha.h"
#include "vf.h"

void
check_gradient (
    BSPLINE_Xform* bxf, 
    BSPLINE_Parms *parms, 
    Volume *fixed, 
    Volume *moving, 
    Volume *moving_grad)
{
    int i, j;
    const float STEP_SIZE = 1e-4;
    float *x, *grad, *grad_fd;
    float score;
    Bspline_state *bst;
    FILE *fp;

    bst = bspline_state_create (bxf);
    x = (float*) malloc (sizeof(float) * bxf->num_coeff);
    grad = (float*) malloc (sizeof(float) * bxf->num_coeff);
    grad_fd = (float*) malloc (sizeof(float) * bxf->num_coeff);

    fp = fopen ("check_grad.txt", "w");

    /* Save a copy of x */
    for (i = 0; i < bxf->num_coeff; i++) {
	x[i] = bxf->coeff[i];
    }

    if (parms->metric == BMET_MI) {
	bspline_initialize_mi (parms, fixed, moving);
    }

    /* Get score and gradient */
    bspline_score (parms, bst, bxf, fixed, moving, moving_grad);

    /* Save a copy of score and gradient */
    for (i = 0; i < bxf->num_coeff; i++) {
	grad[i] = bst->ssd.grad[i];
    }
    score = bst->ssd.score;

    /* Loop through directions */
    for (i = 0; i < bxf->num_coeff; i++) {
	/* Take a step in this direction */
	for (j = 0; j < bxf->num_coeff; j++) {
	    bxf->coeff[j] = x[j];
	}
	bxf->coeff[i] = bxf->coeff[i] + STEP_SIZE;

	/* Get score */
	bspline_score (parms, bst, bxf, fixed, moving, moving_grad);
	
	/* Stash score difference in grad_fd */
	grad_fd[i] = (score - bst->ssd.score) / STEP_SIZE;

	/* Compute difference between grad and grad_fd */
	fprintf (fp, "%10.5f %10.5f\n", grad[i], grad_fd[i]);
    }

    fclose (fp);
    free (x);
    free (grad);
    free (grad_fd);
    bspline_state_destroy (bst);
}

int
main (int argc, char* argv[])
{
    BSPLINE_Options options;
    BSPLINE_Parms* parms = &options.parms;
    BSPLINE_Xform bxf;
    Volume *moving, *fixed, *moving_grad;
    Volume *vector_field = 0;
    Volume *moving_warped = 0;
    int roi_offset[3];

    bspline_opts_parse_args (&options, argc, argv);

    fixed = read_mha (options.fixed_fn);
    if (!fixed) exit (-1);
    moving = read_mha (options.moving_fn);
    if (!moving) exit (-1);

    volume_convert_to_float (moving);
    volume_convert_to_float (fixed);

    printf ("Making gradient\n");
    moving_grad = volume_make_gradient (moving);

    /* Debug */
    //write_mha ("moving_grad.mha", moving_grad);

    /* Allocate memory and build lookup tables */
    printf ("Allocating lookup tables\n");
    memset (roi_offset, 0, 3*sizeof(int));
    bspline_xform_initialize (&bxf,
			      fixed->offset,
			      fixed->pix_spacing,
			      fixed->dim,
			      roi_offset,
			      fixed->dim,
			      options.vox_per_rgn
			     );

    /* Check the gradient */
    check_gradient (&bxf, parms, fixed, moving, moving_grad);

    /* Save output transform */
    if (options.output_xf_fn) {
	write_bxf (options.output_xf_fn, &bxf);
    }

    /* Create vector field from bspline coefficients and save */
    if (options.output_vf_fn || options.output_warped_fn) {
	printf ("Creating vector field.\n");
	vector_field = volume_create (fixed->dim, fixed->offset, 
				      fixed->pix_spacing,
				      PT_VF_FLOAT_INTERLEAVED, 
				      fixed->direction_cosines, 0);
	bspline_interpolate_vf (vector_field, &bxf);
	if (options.output_vf_fn) {
	    printf ("Writing vector field.\n");
	    write_mha (options.output_vf_fn, vector_field);
	}
    }
	
    //printf("%f",moving_warped->dim[1]);
    /* Create warped output image and save */
    if (options.output_warped_fn) {
	printf ("Warping image.\n");
	moving_warped = vf_warp (0, moving, vector_field);
	printf ("Writing warped image.\n");
	//printf("write to %s\n",options.output_warped_fn);
	//system("pause");
	write_mha (options.output_warped_fn, moving_warped);
    }

    /* Free memory */
    printf ("Done warping images.\n");
    bspline_parms_free (parms);
    bspline_xform_free (&bxf);
    volume_destroy (fixed);
    volume_destroy (moving);
    volume_destroy (moving_grad);
    volume_destroy (moving_warped);
    volume_destroy (vector_field);

    printf ("Done freeing memory\n");

    return 0;
}
