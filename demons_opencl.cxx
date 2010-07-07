/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plm_config.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <CL/cl.h>
#include "demons_opts.h"
#include "demons_misc.h"
#include "mha_io.h"
#include "print_and_exit.h"
#include "timer.h"
#include "volume.h"

Volume*
demons_opencl (
    Volume* fixed, 
    Volume* moving, 
    Volume* moving_grad, 
    Volume* vf_init, 
    DEMONS_Parms* parms)
{
    cl_int status = 0;
    cl_uint num_platforms;
    status = clGetPlatformIDs (0, NULL, &num_platforms);
    if (status != CL_SUCCESS) {
	print_and_exit ("Error in clGetPlatformIDs\n");
    }
    if (num_platforms > 0) {
        unsigned int i;
        cl_platform_id* platforms = (cl_platform_id*) malloc (
	    sizeof (cl_platform_id) * num_platforms);
        status = clGetPlatformIDs (num_platforms, platforms, NULL);
	if (status != CL_SUCCESS) {
	    print_and_exit ("Error in clGetPlatformIDs\n");
	}
	
        for (i = 0; i < num_platforms; i++) {
	    char pbuff[100];
            status = clGetPlatformInfo (
		platforms[i],
		CL_PLATFORM_VENDOR,
		sizeof (pbuff),
		pbuff,
		NULL);
	    printf ("OpenCL platform [%d] = %s\n", i, pbuff);
	}	
	free (platforms);
    }
    exit (0);
    return 0;
}
