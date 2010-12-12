#ifndef _DRR_OPENCL_P_H_
#define _DRR_OPENCL_P_H_

#include "drr_opts.h"
#include "math_util.h"
#include "proj_image.h"
#include "volume.h"

#define MAX_GPU_COUNT 8

struct int2 {
	int x;
	int y;
};

struct int3 {
	int x;
	int y;
	int z;
};

struct int4 {
	int x;
	int y;
	int z;
	int w;
};

struct float2 {
	float x;
	float y;
};

struct float3 {
	float x;
	float y;
	float z;
};

struct float4 {
	float x;
	float y;
	float z;
	float w;
};

struct volume_limit_f {
    /* upper and lower limits of volume, including tolerances */
    float lower_limit[3];
    float upper_limit[3];

    /* dir == 0 if lower_limit corresponds to lower index */
    int dir[3];
};
typedef struct volume_limit_f Volume_limit_f;

#endif
