/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plm_config.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#if (OPENMP_FOUND)
#include <omp.h>
#endif
#include "bragg_curve_opts.h"
#include "plm_fortran.h"

// PBDV(V,X,DV,DP,PDF,PDD)
void
pbdv_ (
    doublereal* v,
    doublereal* x,
    doublereal* dv,
    doublereal* dp,
    doublereal* pdf,
    doublereal* pdd);

int
main (int argc, char* argv[])
{
    Bragg_curve_options options;
    doublereal v, x, dv, dp, pdf, pdd;

    //parse_args (&options, argc, argv);

    v = 1;
    x = 1;
    pbdv_ (&v, &x, &dv, &dp, &pdf, &pdd);

    printf ("PCF_(%f) (%f) = (%f %f %f %f)\n",
	v, x, dv, dp, pdf, pdd);

    printf ("Done.\n");
    return 0;
}
