/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plm_config.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <math.h>
#if (NLOPT_FOUND)
#include "nlopt.h"
#endif

#include "bspline.h"
#include "bspline_optimize.h"
#include "bspline_optimize_nlopt.h"
#include "bspline_opts.h"
#include "logfile.h"
#include "math_util.h"
#include "print_and_exit.h"
#include "volume.h"

#if (NLOPT_FOUND)
/* NLopt score function */
static double
bspline_optimize_nlopt_score (
    int n, 
    const double *x, 
    double *grad, 
    void *data)
{
    int i;
    Bspline_optimize_data *bod = (Bspline_optimize_data*) data;
    
    /* Copy x in */
    for (i = 0; i < bod->bxf->num_coeff; i++) {
	bod->bxf->coeff[i] = (float) x[i];
    }

    /* Compute cost and gradient */
    bspline_score (bod);

    /* Copy gradient out */
    for (i = 0; i < bod->bxf->num_coeff; i++) {
	grad[i] = (double) bod->bst->ssd.grad[i];
    }

    /* Return cost */
    return (double) bod->bst->ssd.score;
}

void
bspline_optimize_nlopt (Bspline_optimize_data *bod, nlopt_algorithm algorithm)
{
    int i;
    double *lb, *ub, *x;
    double minf;

    x = (double*) malloc (sizeof(double) * bod->bxf->num_coeff);
    lb = (double*) malloc (sizeof(double) * bod->bxf->num_coeff);
    ub = (double*) malloc (sizeof(double) * bod->bxf->num_coeff);

    for (i = 0; i < bod->bxf->num_coeff; i++) {
	lb[i] = -HUGE_VAL;
	ub[i] = +HUGE_VAL;
	x[i] = (double) bod->bxf->coeff[i];
    }

    nlopt_result nr = 
	nlopt_minimize (
	    algorithm, 
	    bod->bxf->num_coeff, 
	    bspline_optimize_nlopt_score, 
	    bod, 
	    lb, 
	    ub, 
	    x, 
	    &minf, 
	    -HUGE_VAL, 
	    0, 
	    1., 
	    0, 
	    0, 
	    bod->parms->max_its, 
	    0);
    printf ("nlopt returned: %d\n", nr);

    for (i = 0; i < bod->bxf->num_coeff; i++) {
	bod->bxf->coeff[i] = (float) x[i];
    }

    free (x);
    free (ub);
    free (lb);
}

#endif
