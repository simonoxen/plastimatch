/*
 * MEXGPYR_IMG.CPP
 * 
 * Select a subimage from the GPYR
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
   [A_LEV,AW_LEV] = mexgpyr_img(a_gpyr,aw_gpyr,level);
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
    verify_mex_nargs ("mexgpyr_img", nlhs, nrhs, 3, 3, 2, 2);

    /* Check for proper input argument types */
    verify_mex_args_rda (nrhs, prhs);

    /* Extract pointers and matrices from input arguments */
    int level;
    Gpyr *a_gpyr, *aw_gpyr;

    a_gpyr = (Gpyr*) unbundle_pointer_for_matlab (prhs[0]);
    aw_gpyr = (Gpyr*) unbundle_pointer_for_matlab (prhs[1]);
    level = (int) (*mxGetPr(prhs[2]));
    
    /* Allocate memory for output images */
    /* Note: Image::Dims(0) == C width/cols == matlab height/rows */
    Image<GrayScale, double> a = (*a_gpyr)[level];
    Image<GrayScale, double> aw = (*aw_gpyr)[level];
    int rows = a.Dims(0);
    int cols = a.Dims(1);
    int size = rows*cols*sizeof(double);
    mxArray* a_lev = mxCreateDoubleMatrix(rows,cols,mxREAL);
    mxArray* aw_lev = mxCreateDoubleMatrix(rows,cols,mxREAL);
    memcpy(mxGetPr(a_lev),a.Data(),size);
    memcpy(mxGetPr(aw_lev),aw.Data(),size);

    plhs[0] = a_lev;
    plhs[1] = aw_lev;

    return;
}
