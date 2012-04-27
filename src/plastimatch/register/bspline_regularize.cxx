/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plm_config.h"

#include "plmsys.h"

#include "bspline.h"
#include "bspline_regularize_analytic.h"
#include "bspline_regularize_numeric.h"
#include "bspline_xform.h"

void
bspline_regularize_initialize (
    Reg_parms* reg_parms,
    Bspline_regularize_state* rst,
    Bspline_xform* bxf
)
{
    switch (reg_parms->implementation) {
    case 'a':
	bspline_regularize_numeric_a_init (rst, bxf);
        break;
    case 'b':
    case 'c':
        vf_regularize_analytic_init (rst, bxf);
        break;
    case 'd':
	bspline_regularize_numeric_d_init (rst, bxf);
        break;
    default:
        print_and_exit (
            "Error: unknown reg_parms->implementation (%c)\n",
            reg_parms->implementation
        );
        break;
    }
}

void
bspline_regularize_destroy (
    Reg_parms* reg_parms,
    Bspline_regularize_state* rst,
    Bspline_xform* bxf
)
{
    switch (reg_parms->implementation) {
    case 'a':
	bspline_regularize_numeric_a_destroy (rst, bxf);
        break;
    case 'b':
    case 'c':
        vf_regularize_analytic_destroy (rst);
        break;
    case 'd':
	bspline_regularize_numeric_d_destroy (rst, bxf);
        break;
    default:
        print_and_exit (
            "Error: unknown reg_parms->implementation (%c)\n",
            reg_parms->implementation
        );
        break;
    }
}

void
bspline_regularize (
    Bspline_score *bspline_score,    /* Gets updated */
    Bspline_regularize_state* rst,
    const Reg_parms* reg_parms,
    const Bspline_xform* bxf
)
{
    switch (reg_parms->implementation) {
    case 'a':
        bspline_regularize_numeric_a (bspline_score, reg_parms, rst, bxf);
        break;
    case 'b':
        vf_regularize_analytic (bspline_score, reg_parms, rst, bxf);
        break;
    case 'c':
#if (OPENMP_FOUND)
        vf_regularize_analytic_omp (bspline_score, reg_parms, rst, bxf);
#else
        vf_regularize_analytic (bspline_score, reg_parms, rst, bxf);
#endif
        break;
    case 'd':
        bspline_regularize_numeric_d (bspline_score, reg_parms, rst, bxf);
        break;
    default:
        print_and_exit (
            "Error: unknown reg_parms->implementation (%c)\n",
            reg_parms->implementation
        );
        break;
    }
}
