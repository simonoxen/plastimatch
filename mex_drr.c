#include "plm_config.h"
#include "mex.h"

#include "file_util.h"
     
void
mexFunction (int nlhs, mxArray *plhs[], int nrhs, 
    const mxArray *prhs[])
{
    mxArray *v = mxCreateDoubleMatrix (2, 2, mxREAL);
    double *data = mxGetPr (v);
    mxArray *b = prhs[0];
    double *b_data = mxGetPr (b);
    data[0] = b_data[0];
    data[1] = b_data[1];
    data[2] = b_data[2];
    data[3] = b_data[3];

    /* This is just to test linking against the plastimatch library */
    if (extension_is ("foo.pgm", "pgm")) {
	data[3] = 100;
    }
    
    plhs[0] = v;
}
