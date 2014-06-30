/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plmregister_config.h"

#include "bspline_regularize.h"
#include "bspline_regularize_analytic.h"
#include "bspline_regularize_numeric.h"
#include "bspline_xform.h"
#include "print_and_exit.h"

void
Bspline_regularize::initialize (
    Reg_parms* reg_parms,
    Bspline_xform* bxf
)
{
    switch (reg_parms->implementation) {
    case 'a':
	bspline_regularize_numeric_a_init (this, bxf);
        break;
    case 'b':
    case 'c':
        this->vf_regularize_analytic_init (bxf);
        break;
    case 'd':
	bspline_regularize_numeric_d_init (this, bxf);
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
Bspline_regularize::destroy (
    Reg_parms* reg_parms,
    Bspline_xform* bxf
)
{
    switch (reg_parms->implementation) {
    case 'a':
	bspline_regularize_numeric_a_destroy (this, bxf);
        break;
    case 'b':
    case 'c':
        this->vf_regularize_analytic_destroy ();
        break;
    case 'd':
	bspline_regularize_numeric_d_destroy (this, bxf);
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
Bspline_regularize::compute_score (
    Bspline_score *bspline_score,    /* Gets updated */
    const Reg_parms* reg_parms,
    const Bspline_xform* bxf
)
{
    switch (reg_parms->implementation) {
    case 'a':
        bspline_regularize_numeric_a (bspline_score, reg_parms, this, bxf);
        break;
    case 'b':
        vf_regularize_analytic (bspline_score, reg_parms, this, bxf);
        break;
    case 'c':
#if (OPENMP_FOUND)
        vf_regularize_analytic_omp (bspline_score, reg_parms, this, bxf);
#else
        vf_regularize_analytic (bspline_score, reg_parms, this, bxf);
#endif
        break;
    case 'd':
        bspline_regularize_numeric_d (bspline_score, reg_parms, this, bxf);
        break;
    default:
        print_and_exit (
            "Error: unknown reg_parms->implementation (%c)\n",
            reg_parms->implementation
        );
        break;
    }
}
