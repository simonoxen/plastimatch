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
#include "bspline_optimize_steepest.h"
#include "bspline_optimize_lbfgsb.h"
#include "bspline_opts.h"
#include "logfile.h"
#include "math_util.h"
#include "print_and_exit.h"
#include "volume.h"
#include "xpm.h"

void
bspline_optimize (
    BSPLINE_Xform* bxf, 
    Bspline_state *bst, 
    BSPLINE_Parms *parms, 
    Volume *fixed, 
    Volume *moving, 
    Volume *moving_grad)
{
    if (parms->optimization == BOPT_LBFGSB) {
#if (FORTRAN_FOUND)
	bspline_optimize_lbfgsb (bxf, bst, parms, fixed, moving, moving_grad);
#else
	logfile_printf (
	    "LBFGSB not compiled for this platform (no fortran compiler, "
	    "no f2c library).\n  Reverting to steepest descent.\n"
	);
	bspline_optimize_steepest (bxf, bst, parms, fixed, moving, moving_grad);
#endif
    } else {
	bspline_optimize_steepest (bxf, bst, parms, fixed, moving, moving_grad);
    }
}
