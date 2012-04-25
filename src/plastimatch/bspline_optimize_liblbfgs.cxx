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

#include "plmsys.h"

#include "bspline.h"
#include "bspline_optimize.h"
#include "bspline_optimize_liblbfgs.h"
#include "bspline_opts.h"

/* EXTERNAL DEPENDS */
#include "bspline_xform.h"
#include "math_util.h"
#include "print_and_exit.h"
#include "volume.h"
#include "xpm.h"


static lbfgsfloatval_t 
evaluate (
    void *instance,
    const lbfgsfloatval_t *x,
    lbfgsfloatval_t *g,
    const int n,
    const lbfgsfloatval_t step)
{
    Bspline_optimize_data *bod = (Bspline_optimize_data*) instance;
    int i;
    
    /* Copy x in */
    for (i = 0; i < bod->bxf->num_coeff; i++) {
	bod->bxf->coeff[i] = (float) x[i];
    }

    /* Compute cost and gradient */
    bspline_score (bod);

    /* Copy gradient out */
    for (i = 0; i < bod->bxf->num_coeff; i++) {
	g[i] = (lbfgsfloatval_t) bod->bst->ssd.grad[i];
    }

    /* Increment num function evals */
    bod->bst->feval ++;

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
    int n,                        /* Size of input vector */
    int k,                        /* Iteration number */
    int ls                        /* feval within this iteration */
)
{
    Bspline_optimize_data *bod = (Bspline_optimize_data*) instance;

    logfile_printf (
	"                      XN %9.3f GN %9.3f ST %9.3f\n", 
	xnorm, gnorm, step);
    bod->bst->it = k;
    if (bod->bst->it > bod->parms->max_its
	|| bod->bst->feval > bod->parms->max_feval) {
	return 1;
    }
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

    (void) rc;  /* Suppress compiler warning */

    lbfgs_free (x);
}
