/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _drr_cuda_p_h_
#define _drr_cuda_p_h_

#include "drr_cuda.h"

typedef struct drr_kernel_args Drr_kernel_args;
struct drr_kernel_args
{
    int2 img_dim;
    float2 ic;
    float3 nrm;
    float sad;
    float sid;
    float scale;
    float3 vol_offset;
    int3 vol_dim;
    float3 vol_pix_spacing;
    float matrix[12];
    char padding[4]; //for data alignment
    //padding to 128Bytes
};

#endif
