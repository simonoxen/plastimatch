/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _fdk_cuda_h_
#define _fdk_cuda_h_

struct kernel_args_fdk
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
