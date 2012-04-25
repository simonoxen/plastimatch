/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plm_config.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "plmbase.h"

#include "bspline.h"
#include "bspline_mi.h"
#include "bspline_optimize.h"
#if defined (HAVE_F2C_LIBRARY)
#include "bspline_optimize_lbfgsb.h"
#endif
#include "check_grad_opts.h"
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

    FILE *fp;

    plm_long roi_offset[3];

    Bspline_optimize_data bod;
    Bspline_state *bst = bod.bst;
    Bspline_xform *bxf = bod.bxf;
    Bspline_parms *parms = bod.parms;

    parms = &options->parms;
    parms->fixed = fixed;
    parms->moving = moving;
    parms->moving_grad = moving_grad;

    /* Allocate memory and build lookup tables */
    printf ("Allocating lookup tables\n");
    memset (roi_offset, 0, 3*sizeof(plm_long));
    if (options->input_xf_fn) {
        bxf = bspline_xform_load (options->input_xf_fn);
    } else {
        bxf = (Bspline_xform*) malloc (sizeof (Bspline_xform));
        bspline_xform_initialize (
            bxf,
            fixed->offset,
            fixed->spacing,
            fixed->dim,
            roi_offset,
            fixed->dim,
            options->vox_per_rgn,
        (fixed->direction_cosines).m_direction_cosines
        );
        if (options->random) {
            srand (time (0));
            for (i = 0; i < bxf->num_coeff; i++) {
                bxf->coeff[i] = options->random_range[0]
                    + (options->random_range[1] - options->random_range[0])
                    * rand () / (double) RAND_MAX;
            }
        }
    }
    bst = bspline_state_create (bxf, parms);

    /* Create scratch variables */
    x = (float*) malloc (sizeof(float) * bxf->num_coeff);
    grad = (float*) malloc (sizeof(float) * bxf->num_coeff);
    grad_fd = (float*) malloc (sizeof(float) * bxf->num_coeff);

    /* Save a copy of x */
    for (i = 0; i < bxf->num_coeff; i++) {
        x[i] = bxf->coeff[i];
    }

    if (parms->metric == BMET_MI) {
        bspline_initialize_mi (parms);
    }

    /* Get score and gradient */
    bspline_score (&bod);

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
                fprintf (fp, "%4d, %12.12f\n", i, score);
                continue;
            }

            /* Create new location for x */
            for (j = 0; j < bxf->num_coeff; j++) {
                bxf->coeff[j] = x[j] + i * options->step_size * grad[j];
            }

            /* Get score */
            bspline_score (&bod);
        
            /* Compute difference between grad and grad_fd */
            fprintf (fp, "%4d, %12.12f\n", i, bst->ssd.score);

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
            /* Take a step in this direction */
            for (j = 0; j < bxf->num_coeff; j++) {
                bxf->coeff[j] = x[j];
            }
            bxf->coeff[i] = bxf->coeff[i] + options->step_size;

            /* Get score */
            bspline_score (&bod);
        
            /* Stash score difference in grad_fd */
            grad_fd[i] = (bst->ssd.score - score) / options->step_size;

            /* Compute difference between grad and grad_fd */
            fprintf (fp, "%12.12f %12.12f\n", grad[i], grad_fd[i]);
        }
    }

    fclose (fp);
    free (x);
    free (grad);
    free (grad_fd);
    bspline_state_destroy (bst, parms, bxf);
    bspline_xform_free (bxf);
    free (bxf);
    bspline_parms_free (parms);
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
    delete fixed;
    delete moving;
    delete moving_grad;

    printf ("Done freeing memory\n");

    return 0;
}
