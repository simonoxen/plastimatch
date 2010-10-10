/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plm_config.h"

#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include "cuda_util.h"

void cuda_utils_check_error (const char *msg)
{
    cudaError_t err = cudaGetLastError();
    if (cudaSuccess != err)
    {
        fprintf (stderr, "CUDA ERROR: %s (%s).\n", 
	    msg, cudaGetErrorString(err));
        exit (EXIT_FAILURE);
    }                         
}
