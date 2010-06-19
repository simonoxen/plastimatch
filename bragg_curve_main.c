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

double
bragg_curve (double z)
{
    doublereal v, x, dv[100], dp[100], pdf, pdd;

    /* Note: cm, not mm! */
    double p = 1.77;
    double alpha = 0.0022;
    double R_0 = 20.0;
    double E_0 = R_0 / alpha;
    double beta = 0.012;
    double gamma = 0.6;
    double sigma_mono = 0.012 * pow (R_0, 0.935);
    double sigma_E0 = 0.01 * E_0;
    double epsilon = 0.1;

    double sigma_2 = sigma_mono * sigma_mono 
	+ (sigma_E0 * sigma_E0 * alpha * alpha * p * p 
	    * pow (E_0 * E_0, (p - 2)));
    double sigma = sqrt (sigma_2);
    
    double rr = R_0 - z;  /* rr = residual range */

    double bragg;

    //printf ("sigma = (%f,%f,%f)\n", sigma_mono, sigma_E0, sigma);

    //sigma = sigma_E0;

    /* Use Dhat */
    if (rr > 10 * sigma) {
	bragg = 1 / (1 + 0.012 * R_0) 
	    * (17.93 * pow (rr, -0.435)
		+ (0.444 + 31.7 * epsilon / R_0) * pow (rr, 0.565));
	return bragg;
    }

    /* Term 1 of eqn 29 */
    bragg = exp (- (rr * rr) / (4 * sigma * sigma)) * pow (sigma, 0.565)
	/ (1 + 0.012 * R_0);

    /* D_v in term 2 */
    v = - 0.565;
    x = - rr / sigma;
    printf ("x = %f\n", x);
    pbdv_ (&v, &x, dv, dp, &pdf, &pdd);
#if defined (commentout)
    printf ("PCF_(%f) (%f) = (%f %f %f %f)\n",
	v, x, dv, dp, pdf, pdd);
#endif
    
    /* Term 2 of eqn 29 */
    bragg = bragg * (11.26 / sigma) * pdf;

    /* D_v in term 3 */
    v = - 1.565;
    x = - rr / sigma;
    pbdv_ (&v, &x, dv, dp, &pdf, &pdd);
#if defined (commentout)
    printf ("PCF_(%f) (%f) = (%f %f %f %f)\n",
	v, x, dv, dp, pdf, pdd);
#endif
    
    /* term 3 of eqn 29 */
    bragg = bragg + (0.157 + 11.26 * epsilon / R_0) * pdf;

    //return bragg;
    return x;
}

int
main (int argc, char* argv[])
{
    Bragg_curve_options options;
    double z;

    //parse_args (&options, argc, argv);
    //printf ("sigma = %f\n", sigma);

    for (z = 15.00; z < 30.0; z += 1) {
	printf ("%f %f\n", z, bragg_curve (z));
    }

    printf ("Done.\n");
    return 0;
}
