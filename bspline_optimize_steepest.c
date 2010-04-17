/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plm_config.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <math.h>
#if (OPENMP_FOUND)
#include <omp.h>
#endif

#include "bspline.h"
#if (CUDA_FOUND)
#include "bspline_cuda.h"
#endif
#include "bspline_optimize_steepest.h"
#include "bspline_opts.h"
#include "logfile.h"
#include "math_util.h"
#include "print_and_exit.h"
#include "readmha.h"
#include "timer.h"
#include "volume.h"
#include "xpm.h"

/* This combines a steepest descent direction with trust interval line search.
   See Eqn 2.8 + Eqn 2.20 in Madsen, Nielsen, and Tingleff's 
     booklet: "Methods for non-linear least squares probelms"
     http://www2.imm.dtu.dk/pubdb/views/edoc_download.php/3215/pdf/imm3215.pdf
   See also: http://www2.imm.dtu.dk/~hbn/immoptibox/ 

   It works ok, but seems to get caught in Stiefel's cage.
*/
void
bspline_optimize_steepest_trust (
    BSPLINE_Xform *bxf, 
    Bspline_state *bst, 
    BSPLINE_Parms *parms, 
    Volume *fixed, 
    Volume *moving, 
    Volume *moving_grad
)
{
    BSPLINE_Score* ssd = &bst->ssd;
    int i;
    float alpha = 1.0f;
    float ssd_grad_norm;
    float old_score;
    FILE *fp;
    float *x;           /* Start of line search */
    float *h;           /* Search direction */
    double htg;

    if (parms->debug) {
	fp = fopen("scores.txt", "w");
    }

    /* Allocate memory for search direction */
    x = (float*) malloc (bxf->num_coeff * sizeof(float));
    h = (float*) malloc (bxf->num_coeff * sizeof(float));

    /* Set iteration */
    bst->it = 0;
    memcpy (x, bxf->coeff, bxf->num_coeff * sizeof(float));

    /* Get score and gradient */
    bspline_score (parms, bst, bxf, fixed, moving, moving_grad);
    old_score = bst->ssd.score;

    /* Get search direction */
    ssd_grad_norm = 0;
    for (i = 0; i < bxf->num_coeff; i++) {
	ssd_grad_norm += ssd->grad[i] * ssd->grad[i];
    }
    ssd_grad_norm = sqrt (ssd_grad_norm);
    htg = 0.0;
    for (i = 0; i < bxf->num_coeff; i++) {
#if PLM_DONT_INVERT_GRADIENT
	h[i] = - ssd->grad[i] / ssd_grad_norm;
	htg -= h[i] * ssd->grad[i];
#else
	h[i] = ssd->grad[i] / ssd_grad_norm;
	htg += h[i] * ssd->grad[i];
#endif
    }

    /* Give a little feedback to the user */
    bspline_display_coeff_stats (bxf);
    /* Save some debugging information */
    bspline_save_debug_state (parms, bst, bxf);
    if (parms->debug) {
	fprintf (fp, "%f\n", ssd->score);
    }

    while (bst->it < parms->max_its) {
	double gr;
	int accept_step = 0;

	/* Update iteration number */
	bst->it ++;

	/* Compute new search location */
	for (i = 0; i < bxf->num_coeff; i++) {
	    bxf->coeff[i] = x[i] + alpha * h[i];
	}

	/* Get score and gradient */
	bspline_score (parms, bst, bxf, fixed, moving, moving_grad);

	/* Compute gain ratio with linear model */
	gr = (old_score - bst->ssd.score) / htg;
	if (gr < 0) {
	    /* Cost increased.  Keep search direction and reduce trust rgn. */
	    alpha = 0.5 * alpha;
	} else if (gr < 0.25) {
	    alpha = 0.5 * alpha;
	    accept_step = 1;
	} else if (gr > 0.75) {
	    alpha = 3.0 * alpha;
	    accept_step = 1;
	} else {
	    accept_step = 1;
	}

	/* Give a little feedback to the user */
	bspline_display_coeff_stats (bxf);
	logfile_printf ("                    "
	    "GR %6.2f NEW_A %6.2f ACCEPT? %d\n", gr, alpha, accept_step);

	/* Save some debugging information */
	bspline_save_debug_state (parms, bst, bxf);
	if (parms->debug) {
	    fprintf (fp, "%f\n", ssd->score);
	}

	/* If score was reduced, we accept the line search */
	if (!accept_step) continue;

	/* Start new line search */
	memcpy (x, bxf->coeff, bxf->num_coeff * sizeof(float));
	ssd_grad_norm = 0;
	for (i = 0; i < bxf->num_coeff; i++) {
	    ssd_grad_norm += ssd->grad[i] * ssd->grad[i];
	}
	ssd_grad_norm = sqrt (ssd_grad_norm);
	htg = 0.0;
	for (i = 0; i < bxf->num_coeff; i++) {
#if PLM_DONT_INVERT_GRADIENT
	    h[i] = - ssd->grad[i] / ssd_grad_norm;
	    htg -= h[i] * ssd->grad[i];
#else
	    h[i] = ssd->grad[i] / ssd_grad_norm;
	    htg += h[i] * ssd->grad[i];
#endif
	}
	old_score = bst->ssd.score;
    }

    /* Save best result */
    memcpy (bxf->coeff, x, bxf->num_coeff * sizeof(float));
    bst->ssd.score = old_score;

    if (parms->debug) {
	fclose (fp);
    }
    
    free (x);
    free (h);
}

/* This is a really terrible algorithm.  It takes steps without 
   doing any sort of line search. */
void
bspline_optimize_steepest_naive (
    BSPLINE_Xform *bxf, 
    Bspline_state *bst, 
    BSPLINE_Parms *parms, 
    Volume *fixed, 
    Volume *moving, 
    Volume *moving_grad
)
{
    BSPLINE_Score* ssd = &bst->ssd;
    int i;
    //    float a = 0.003f;
    //    float alpha = 0.5f, A = 10.0f;
    float a, gamma;
    float gain = 1.5;
    float ssd_grad_norm;
    float old_score;
    FILE* fp;

    if (parms->debug) {
	fp = fopen("scores.txt", "w");
    }

    /* Set iteration */
    bst->it = 0;

    /* Get score and gradient */
    bspline_score (parms, bst, bxf, fixed, moving, moving_grad);
    old_score = bst->ssd.score;

    /* Set alpha based on norm gradient */
    ssd_grad_norm = 0;
    for (i = 0; i < bxf->num_coeff; i++) {
	ssd_grad_norm += fabs (ssd->grad[i]);
    }
    a = 1.0f / ssd_grad_norm;
    gamma = a;
    logfile_printf ("Initial gamma is %g\n", gamma);

    /* Give a little feedback to the user */
    bspline_display_coeff_stats (bxf);
    /* Save some debugging information */
    bspline_save_debug_state (parms, bst, bxf);
    if (parms->debug) {
	fprintf (fp, "%f\n", ssd->score);
    }

    while (bst->it < parms->max_its) {

	/* Update iteration number */
	bst->it ++;

	logfile_printf ("Beginning iteration %d, gamma = %g\n", bst->it, gamma);

	/* Update b-spline coefficients from gradient */
	//gamma = a / pow(it + A, alpha);
	for (i = 0; i < bxf->num_coeff; i++) {
	    bxf->coeff[i] = bxf->coeff[i] + gamma * ssd->grad[i];
	}

	/* Get score and gradient */
	bspline_score (parms, bst, bxf, fixed, moving, moving_grad);

	/* Update gamma */
	if (bst->ssd.score < old_score) {
	    gamma *= gain;
	} else {
	    gamma /= gain;
	}
	old_score = bst->ssd.score;

	/* Give a little feedback to the user */
	bspline_display_coeff_stats (bxf);
	/* Save some debugging information */
	bspline_save_debug_state (parms, bst, bxf);
	if (parms->debug) {
	    fprintf (fp, "%f\n", ssd->score);
	}
    }

    if (parms->debug) {
	fclose (fp);
    }
}

void
bspline_optimize_steepest (
    BSPLINE_Xform *bxf, 
    Bspline_state *bst, 
    BSPLINE_Parms *parms, 
    Volume *fixed, 
    Volume *moving, 
    Volume *moving_grad
)
{
    const int USE_NAIVE = 0;

    if (USE_NAIVE) {
	bspline_optimize_steepest_naive (
	    bxf, bst, parms, fixed, moving, moving_grad);
    } else {
	bspline_optimize_steepest_trust (
	    bxf, bst, parms, fixed, moving, moving_grad);
    }
}
