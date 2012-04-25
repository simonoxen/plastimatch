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
#include "bspline_mi.h"
#include "bspline_optimize.h"
#include "bspline_optimize_liblbfgs.h"
#include "bspline_optimize_lbfgsb.h"
#if (NLOPT_FOUND)
#include "bspline_optimize_nlopt.h"
#endif
#include "bspline_optimize_steepest.h"
#include "bspline_opts.h"

/* EXTERNAL DEPENDS */
#include "bspline_xform.h"
#include "logfile.h"
#include "math_util.h"
#include "print_and_exit.h"
#include "volume.h"
#include "xpm.h"

static void
log_parms (Bspline_parms* parms)
{
    logfile_printf ("BSPLINE PARMS\n");
    logfile_printf ("max_its = %d\n", parms->max_its);
    logfile_printf ("max_feval = %d\n", parms->max_feval);
}

static void
log_bxf_header (Bspline_xform* bxf)
{
    logfile_printf ("BSPLINE XFORM HEADER\n");
    logfile_printf ("vox_per_rgn = %d %d %d\n", bxf->vox_per_rgn[0], bxf->vox_per_rgn[1], bxf->vox_per_rgn[2]);
    logfile_printf ("roi_offset = %d %d %d\n", bxf->roi_offset[0], bxf->roi_offset[1], bxf->roi_offset[2]);
    logfile_printf ("roi_dim = %d %d %d\n", bxf->roi_dim[0], bxf->roi_dim[1], bxf->roi_dim[2]);
}

static void
bspline_optimize_select (
    Bspline_optimize_data *bod
)
{
    Bspline_parms *parms = bod->parms;

    switch (parms->optimization) {
    case BOPT_LBFGSB:
#if (FORTRAN_FOUND)
        //bspline_optimize_lbfgsb (bxf, bst, parms, fixed, moving, moving_grad);
        bspline_optimize_lbfgsb (bod);
#else
        logfile_printf (
            "Plastimatch was not compiled against Nocedal LBFGSB.\n"
            "Reverting to liblbfgs.\n"
        );
        bspline_optimize_liblbfgs (bod);
#endif
        break;
    case BOPT_STEEPEST:
        bspline_optimize_steepest (bod);
        break;
    case BOPT_LIBLBFGS:
        bspline_optimize_liblbfgs (bod);
        break;
#if (NLOPT_FOUND)
    case BOPT_NLOPT_LBFGS:
        bspline_optimize_nlopt (bod, NLOPT_LD_LBFGS);
        break;
    case BOPT_NLOPT_LD_MMA:
        bspline_optimize_nlopt (bod, NLOPT_LD_MMA);
        break;
    case BOPT_NLOPT_PTN_1:
        //bspline_optimize_nlopt (&bod, NLOPT_LD_TNEWTON_PRECOND_RESTART);
        bspline_optimize_nlopt (bod, NLOPT_LD_VAR2);
        break;
#else
    case BOPT_NLOPT_LBFGS:
    case BOPT_NLOPT_LD_MMA:
    case BOPT_NLOPT_PTN_1:
        logfile_printf (
            "Plastimatch was not compiled against NLopt.\n"
            "Reverting to liblbfgs.\n"
        );
        bspline_optimize_liblbfgs (bod);
#endif
    default:
        bspline_optimize_liblbfgs (bod);
        break;
    }
}

void
bspline_optimize (
    Bspline_xform* bxf, 
    Bspline_state **bst_in, 
    Bspline_parms *parms
)
{
    Bspline_optimize_data bod;
    bod.bxf = bxf;
    bod.parms = parms;

#if 0
    parms->fixed = fixed;
    parms->moving = moving;
    parms->moving_grad = moving_grad;
#endif

    bod.bst = bspline_state_create (bxf, parms);
    log_parms (parms);
    log_bxf_header (bxf);

    if (parms->metric == BMET_MI) {
        bspline_initialize_mi (parms);
    }

    /* Do the optimization */
    bspline_optimize_select (&bod);

    if (bst_in) {
        *bst_in = bod.bst;
    } else {
        bspline_state_destroy (bod.bst, parms, bxf);
    }
}
