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

void drr_render_volume_perspective_cl(Proj_image *proj, Volume *vol, double ps[2], Drr_options *options, int img_size, int float3_size, cl_mem *g_dev_vol, cl_mem *g_dev_img, cl_mem *c_vol_dim, cl_mem *c_img_dim, cl_mem *c_offset, cl_mem *c_pix_spacing, cl_mem *c_vol_limits, cl_mem *c_p1, cl_mem *c_ul_room, cl_mem *c_incr_r, cl_mem *c_incr_c, cl_mem *c_pixel_device, cl_kernel *drr_kernel, cl_ulong *drr_total, cl_ulong *img_total, size_t drr_global_work_size[MAX_GPU_COUNT][2], size_t drr_local_work_size[MAX_GPU_COUNT][2], int4 *pixels_per_device, int2 *pixel_offset, int *img_size_device);
void create_matrix_and_drr_cl(Volume* vol, Proj_image *proj, double cam[3], double tgt[3], double nrm[3], int a, Drr_options* options, int img_size, int float3_size, cl_mem *g_dev_vol, cl_mem *g_dev_img, cl_mem *c_vol_dim, cl_mem *c_img_dim, cl_mem *c_offset, cl_mem *c_pix_spacing, cl_mem *c_vol_limits, cl_mem *c_p1, cl_mem *c_ul_room, cl_mem *c_incr_r, cl_mem *c_incr_c, cl_mem *c_pixel_device, cl_kernel *drr_kernel, cl_ulong *drr_total, cl_ulong *img_total, size_t drr_global_work_size[MAX_GPU_COUNT][2], size_t drr_local_work_size[MAX_GPU_COUNT][2], int4 *pixels_per_device, int2 *pixel_offset, int *img_size_device);

#endif