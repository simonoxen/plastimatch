/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plm_config.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <math.h>
#include "lbfgs.h"

#include "bspline.h"
#include "bspline_optimize.h"
#include "bspline_optimize_liblbfgs.h"
#include "bspline_opts.h"
#include "logfile.h"
#include "math_util.h"
#include "print_and_exit.h"
#include "volume.h"

static lbfgsfloatval_t 
evaluate (
    void *instance,
    const lbfgsfloatval_t *x,
    lbfgsfloatval_t *g,
    const int n,
    const lbfgsfloatval_t step)
{
    int i;
    Bspline_optimize_data *bod = (Bspline_optimize_data*) instance;
    
    /* Copy x in */
    for (i = 0; i < bod->bxf->num_coeff; i++) {
	bod->bxf->coeff[i] = (float) x[i];
    }

    /* Compute cost and gradient */
    bspline_score (bod->parms, bod->bst, bod->bxf, bod->fixed, 
	bod->moving, bod->moving_grad);

    /* Copy gradient out */
    for (i = 0; i < bod->bxf->num_coeff; i++) {
#if PLM_DONT_INVERT_GRADIENT
	g[i] = (lbfgsfloatval_t) bod->bst->ssd.grad[i];
#else
	g[i] = - (lbfgsfloatval_t) bod->bst->ssd.grad[i];
#endif
    }

    /* Return cost */
    return (lbfgsfloatval_t) bod->bst->ssd.score;
}

static int
progress(
    void *instance,
    const lbfgsfloatval_t *x,
    const lbfgsfloatval_t *g,
    const lbfgsfloatval_t fx,
    const lbfgsfloatval_t xnorm,
    const lbfgsfloatval_t gnorm,
    const lbfgsfloatval_t step,
    int n,
    int k,
    int ls)
{
    printf("Iteration %d:\n", k);
    printf("  fx = %f, x[0] = %f, x[1] = %f\n", fx, x[0], x[1]);
    printf("  xnorm = %f, gnorm = %f, step = %f\n", xnorm, gnorm, step);
    printf("\n");
    return 0;
}

void
bspline_optimize_liblbfgs (Bspline_optimize_data *bod)
{
    int i, rc;
    lbfgsfloatval_t fx;
    lbfgs_parameter_t param;
    lbfgsfloatval_t *x;

    x = lbfgs_malloc (bod->bxf->num_coeff);

    /* Convert x0 from float to lbfgsfloatval_t */
    for (i = 0; i < bod->bxf->num_coeff; i++) {
	x[i] = (lbfgsfloatval_t) bod->bxf->coeff[i];
    }

    /* Set default parameters */
    lbfgs_parameter_init (&param);

    /* Run the optimizer */
    rc = lbfgs (bod->bxf->num_coeff, x, &fx, 
	evaluate, progress, (void*) bod, &param);

    lbfgs_free (x);
}
