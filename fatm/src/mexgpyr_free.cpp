/*
 * MEXGPYR.CPP
 * 
 * Make the gaussian pyramid and return opaque pointer.
 * 
 */
#include <math.h>
#include <vector>
#include "mex.h"
#include "mexutils.h"
#include "gpyr.h"

extern void _main();

/* -----------------------------------------------------------------------
   This function is called like this:
   mexgpyr_free(a_gpyr,aw_gpyr);
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
    verify_mex_nargs ("mexgpyr", nlhs, nrhs, 1, 2, 0, 0);

    /* Check for proper input argument types */
    verify_mex_args_rda (nrhs, prhs);

    /* Extract pointers from input arguments, and free memory */
    delete (Gpyr*) unbundle_pointer_for_matlab (prhs[0]);
    if (nrhs == 2) {
	delete (Gpyr*) unbundle_pointer_for_matlab (prhs[1]);
    }
    
    return;
}
