/*
 * MEXGPYR.CPP
 * 
 * Make the gaussian pyramid and return opaque pointer.
 * 
 */
#include <math.h>
#include <vector>
#include "mex.h"
#include "scorewin.h"
#include "s_wncc.h"
#include "mexutils.h"
#include "gpyr.h"

extern void _main();

/* -----------------------------------------------------------------------
   This function is called like this:
   [a_gpyr,aw_gpyr] = mexgpyr(A,AW,num_levels);
   ----------------------------------------------------------------------- */
void mexFunction(
		 int          nlhs,
		 mxArray      *plhs[],
		 int          nrhs,
		 const mxArray *prhs[]
		 )
{
    double *outvp;
    double *invp;
    double *outv;
    double *inv;

    /* Check for proper number of arguments */
    verify_mex_nargs ("mexgpyr", nlhs, nrhs, 3, 3, 2, 2);

    /* Check for proper input argument types */
    verify_mex_args_rda (nrhs, prhs);

    /* Extract matrices from input arguments */
    Image<GrayScale,double> a,aw;
    a = Image<GrayScale,double>(mxGetM(prhs[0]),mxGetN(prhs[0]),
				(double*) mxGetPr(prhs[0]));
    aw = Image<GrayScale,double>(mxGetM(prhs[1]),mxGetN(prhs[1]),
				 (double*) mxGetPr(prhs[1]));
    int n_levels = (int) (*mxGetPr(prhs[2]));
    
    /* Allocate memory and build the gaussian pyramid */
    std::vector<Image<GrayScale, double> > *pat_pyr, *pat_mask_pyr;
    pat_pyr = new std::vector<Image<GrayScale, double> >;
    pat_mask_pyr = new std::vector<Image<GrayScale, double> >;
    BuildPyramid<double> (pat_pyr, pat_mask_pyr, a, aw, n_levels);

    plhs[0] = bundle_pointer_for_matlab ((void*) pat_pyr);
    plhs[1] = bundle_pointer_for_matlab ((void*) pat_mask_pyr);

    return;
}
