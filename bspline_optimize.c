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
#include "bspline_optimize_lbfgsb.h"
#include "bspline_optimize_steepest.h"
#include "bspline_opts.h"
#include "logfile.h"
#include "math_util.h"
#include "print_and_exit.h"
#include "volume.h"
#include "xpm.h"

#if (NLOPT_FOUND)
/* NLopt score function */
double
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
    bspline_score (bod->parms, bod->bst, bod->bxf, bod->fixed, 
	bod->moving, bod->moving_grad);

    /* Copy gradient out */
    for (i = 0; i < bod->bxf->num_coeff; i++) {
	grad[i] = (double) bod->bst->ssd.grad[i];
    }

    /* Return cost */
    return (double) bod->bst->ssd.score;
}

void
bspline_optimize_nlopt_lbfgs (Bspline_optimize_data *bod)
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
	    NLOPT_LD_LBFGS, 
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

    for (i = 0; i < bod->bxf->num_coeff; i++) {
	bod->bxf->coeff[i] = (float) x[i];
    }

    free (x);
    free (ub);
    free (lb);
}
#endif

void
bspline_optimize (
    BSPLINE_Xform* bxf, 
    Bspline_state *bst, 
    BSPLINE_Parms *parms, 
    Volume *fixed, 
    Volume *moving, 
    Volume *moving_grad)
{
    Bspline_optimize_data bod;
    bod.bxf = bxf;
    bod.bst = bst;
    bod.parms = parms;
    bod.fixed = fixed;
    bod.moving = moving;
    bod.moving_grad = moving_grad;

    switch (parms->optimization) {
    case BOPT_LBFGSB:
#if (FORTRAN_FOUND)
	bspline_optimize_lbfgsb (bxf, bst, parms, fixed, moving, moving_grad);
#else
	logfile_printf (
	    "LBFGSB not compiled for this platform (no fortran compiler, "
	    "no f2c library).\n  Reverting to steepest descent.\n"
	);
	bspline_optimize_steepest (bxf, bst, parms, fixed, moving, moving_grad);
#endif
	break;
    case BOPT_STEEPEST:
	bspline_optimize_steepest (bxf, bst, parms, fixed, moving, moving_grad);
	break;
#if (NLOPT_FOUND)
    case BOPT_NLOPT_LBFGS:
	bspline_optimize_nlopt_lbfgs (&bod);
	break;
#else
    case BOPT_NLOPT_LBFGS:
	logfile_printf (
	    "LBFGSB not compiled for this platform (no fortran compiler, "
	    "no f2c library).\n  Reverting to steepest descent.\n"
	);
	bspline_optimize_steepest (bxf, bst, parms, fixed, moving, moving_grad);
#endif
    default:
	bspline_optimize_steepest (bxf, bst, parms, fixed, moving, moving_grad);
	break;
    }
}
