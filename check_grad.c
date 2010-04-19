/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plm_config.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "bspline.h"
#if defined (HAVE_F2C_LIBRARY)
#include "bspline_optimize_lbfgsb.h"
#endif
#include "check_grad_opts.h"
#include "readmha.h"
#include "vf.h"

void
check_gradient (
    Check_grad_opts *options, 
    Volume *fixed, 
    Volume *moving, 
    Volume *moving_grad)
{
    int i, j;
    float *x, *grad, *grad_fd;
    float score;
    Bspline_state *bst;
    FILE *fp;
    BSPLINE_Xform *bxf;
    int roi_offset[3];
    BSPLINE_Parms* parms = &options->parms;

    /* Allocate memory and build lookup tables */
    printf ("Allocating lookup tables\n");
    memset (roi_offset, 0, 3*sizeof(int));
    if (options->input_xf_fn) {
	bxf = read_bxf (options->input_xf_fn);
    } else {
	bxf = (BSPLINE_Xform*) malloc (sizeof (BSPLINE_Xform));
	bspline_xform_initialize (
	    bxf,
	    fixed->offset,
	    fixed->pix_spacing,
	    fixed->dim,
	    roi_offset,
	    fixed->dim,
	    options->vox_per_rgn
	);
    }
    bst = bspline_state_create (bxf, parms, fixed, moving, moving_grad);

    /* Create scratch variables */
    x = (float*) malloc (sizeof(float) * bxf->num_coeff);
    grad = (float*) malloc (sizeof(float) * bxf->num_coeff);
    grad_fd = (float*) malloc (sizeof(float) * bxf->num_coeff);

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

    fp = fopen (options->output_fn, "w");
    if (options->process == CHECK_GRAD_PROCESS_LINE) {
	/* For each step along line */
	for (i = options->line_range[0]; i < options->line_range[1]; i++) {
	    bst->it = i;

	    /* Already computed for i = 0 */
	    if (i == 0) {
		fprintf (fp, "%d, %10.10f\n", i, score);
		continue;
	    }

	    /* Create new location for x */
	    for (j = 0; j < bxf->num_coeff; j++) {
		bxf->coeff[j] = x[j] + i * options->step_size * grad[j];
	    }

	    /* Get score */
	    bspline_score (parms, bst, bxf, fixed, moving, moving_grad);
	
	    /* Compute difference between grad and grad_fd */
	    fprintf (fp, "%d, %10.10f\n", i, bst->ssd.score);

	    // JAS 04.19.2010
	    // This loop could take a while to exit.  This will
	    // flush the buffer so that we will at least get the data
	    // that we worked for if we get sick of waiting and opt
	    // for early program termination.
	    fflush(fp);
	}
    } else {
	/* Loop through directions */
	for (i = 0; i < bxf->num_coeff; i++) {
	    /* Take a step in this direction */ for (j = 0; j < bxf->num_coeff; j++) {
		bxf->coeff[j] = x[j];
	    }
	    bxf->coeff[i] = bxf->coeff[i] + options->step_size;

	    /* Get score */
	    bspline_score (parms, bst, bxf, fixed, moving, moving_grad);
	
	    /* Stash score difference in grad_fd */
	    grad_fd[i] = (score - bst->ssd.score) / options->step_size;

	    /* Compute difference between grad and grad_fd */
	    fprintf (fp, "%10.10f %10.10f\n", grad[i], grad_fd[i]);
	}
    }

    fclose (fp);
    free (x);
    free (grad);
    free (grad_fd);
    bspline_xform_free (bxf);
    free (bxf);
    bspline_parms_free (parms);
    bspline_state_destroy (bst);
}

int
main (int argc, char* argv[])
{
    Check_grad_opts options;
    Volume *moving, *fixed, *moving_grad;

    check_grad_opts_parse_args (&options, argc, argv);

    fixed = read_mha (options.fixed_fn);
    if (!fixed) exit (-1);
    moving = read_mha (options.moving_fn);
    if (!moving) exit (-1);

    volume_convert_to_float (moving);
    volume_convert_to_float (fixed);

    printf ("Making gradient\n");
    moving_grad = volume_make_gradient (moving);

    /* Check the gradient */
    check_gradient (&options, fixed, moving, moving_grad);

    /* Free memory */
    volume_destroy (fixed);
    volume_destroy (moving);
    volume_destroy (moving_grad);

    printf ("Done freeing memory\n");

    return 0;
}
