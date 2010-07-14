/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plm_config.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <math.h>

#include "bspline.h"
#include "bspline_optimize.h"
#include "bspline_optimize_liblbfgs.h"
#include "bspline_optimize_lbfgsb.h"
#if (NLOPT_FOUND)
#include "bspline_optimize_nlopt.h"
#endif
#include "bspline_optimize_steepest.h"
#include "bspline_opts.h"
#include "logfile.h"
#include "math_util.h"
#include "print_and_exit.h"
#include "volume.h"
#include "xpm.h"

void
bspline_optimize (
    Bspline_xform* bxf, 
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
	    "Plastimatch was not compiled against Nocedal LBFGSB.\n"
	    "Reverting to liblbfgs.\n"
	);
	bspline_optimize_liblbfgs (&bod);
#endif
	break;
    case BOPT_STEEPEST:
      bspline_optimize_steepest (bxf, bst, parms, fixed, moving, moving_grad);
	break;
    case BOPT_LIBLBFGS:
	bspline_optimize_liblbfgs (&bod);
	break;
#if (NLOPT_FOUND)
    case BOPT_NLOPT_LBFGS:
	bspline_optimize_nlopt (&bod, NLOPT_LD_LBFGS);
	break;
    case BOPT_NLOPT_LD_MMA:
	bspline_optimize_nlopt (&bod, NLOPT_LD_MMA);
	break;
    case BOPT_NLOPT_PTN_1:
	//bspline_optimize_nlopt (&bod, NLOPT_LD_TNEWTON_PRECOND_RESTART);
	bspline_optimize_nlopt (&bod, NLOPT_LD_VAR2);
	break;
#else
    case BOPT_NLOPT_LBFGS:
    case BOPT_NLOPT_LD_MMA:
    case BOPT_NLOPT_PTN_1:
	logfile_printf (
	    "Plastimatch was not compiled against NLopt.\n"
	    "Reverting to liblbfgs.\n"
	);
	bspline_optimize_liblbfgs (&bod);
#endif
    default:
	bspline_optimize_liblbfgs (&bod);
	break;
    }
}
