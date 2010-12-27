/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plm_config.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#if defined (_WIN32)
#include <windows.h>
#endif

#include "fdk.h"
#include "fdk_cuda.h"

void*
fdk_cuda_state_create (
    Volume *vol, 
    unsigned int image_npix, 
    float scale, 
    Fdk_options *options
)
{
    return fdk_cuda_state_create_cu (vol, image_npix, scale, options);
}

void
fdk_cuda_state_destroy (
    void *void_state
)
{
    fdk_cuda_state_destroy_cu (void_state);
}

void
fdk_cuda_queue_image (
    void *dev_state, 
    int *dim, 
    double *ic, 
    double *nrm, 
    double sad, 
    double sid, 
    double *matrix, 
    float *img
)
{
    fdk_cuda_queue_image_cu (dev_state, dim, ic, nrm, sad, sid, 
	matrix, img);
}

void
fdk_cuda_backproject (void *dev_state)
{
    fdk_cuda_backproject_cu (dev_state);
}

void
fdk_cuda_fetch_volume (
    void *dev_state, 
    void *host_buf, 
    unsigned int copy_size
)
{
    fdk_cuda_fetch_volume_cu (dev_state, host_buf, copy_size);
}
