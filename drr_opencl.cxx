/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plm_config.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "drr_opencl.h"
#include "drr_opencl_p.h"
#include "drr.h"
#include "drr_opts.h"
#include "file_util.h"
#include "math_util.h"
#include "opencl_util.h"
#include "opencl_util_nvidia.h"
#include "plm_timer.h"
#include "proj_image.h"
#include "proj_matrix.h"
#include "volume.h"
#include "volume_limit.h"

void
drr_opencl_ray_trace_image (
    Proj_image *proj, 
    Volume *vol, 
    Volume_limit *vol_limit, 
    double p1[3], 
    double ul_room[3], 
    double incr_r[3], 
    double incr_c[3], 
    void *dev_state, 
    Drr_options *options
)
{
    Opencl_device ocl_dev;
    Opencl_buf *ocl_buf_img;
    Opencl_buf *ocl_buf_vol;
    Opencl_buf *ocl_buf_vol_dim;
    Opencl_buf *ocl_buf_vol_offset;
    Opencl_buf *ocl_buf_vol_spacing;
    Opencl_buf *ocl_buf_img_dim;
    Opencl_buf *ocl_buf_ic;
    Opencl_buf *ocl_buf_img_window;
    Opencl_buf *ocl_buf_p1;
    Opencl_buf *ocl_buf_ul_room;
    Opencl_buf *ocl_buf_incr_r;
    Opencl_buf *ocl_buf_incr_c;
    Opencl_buf *ocl_buf_nrm;
    Opencl_buf *ocl_buf_lower_limit;
    Opencl_buf *ocl_buf_upper_limit;

    Proj_matrix *pmat = proj->pmat;
    float tmp[3];
    float sad;

    /* Set up devices and kernels */
    opencl_open_device (&ocl_dev);
    opencl_load_programs (&ocl_dev, "drr_opencl.cl");
    opencl_kernel_create (&ocl_dev, "kernel_drr");

    /* Set up device memory */
    ocl_buf_vol = opencl_buf_create (
        &ocl_dev, 
        CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, 
        vol->pix_size * vol->npix,
        vol->img
    );

    ocl_buf_img = opencl_buf_create (
        &ocl_dev, 
        CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, 
        proj->dim[1] * proj->dim[0] * sizeof(float),
        0
    );

    ocl_buf_vol_dim = opencl_buf_create (
        &ocl_dev, 
        CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, 
        3 * sizeof(int),
        vol->dim
    );

    ocl_buf_vol_offset = opencl_buf_create (
        &ocl_dev, 
        CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, 
        3 * sizeof(float),
        vol->offset
    );

    ocl_buf_vol_spacing = opencl_buf_create (
        &ocl_dev, 
        CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, 
        3 * sizeof(float),
        vol->pix_spacing
    );

    ocl_buf_img_dim = opencl_buf_create (
        &ocl_dev, 
        CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, 
        2 * sizeof(int),
        proj->dim
    );

    ocl_buf_ic = opencl_buf_create (
        &ocl_dev, 
        CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR, 
        2 * sizeof(float),
	0
    );

    ocl_buf_img_window = opencl_buf_create (
        &ocl_dev, 
        CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, 
        4 * sizeof(int),
	options->image_window
    );

    ocl_buf_p1 = opencl_buf_create (
        &ocl_dev, 
        CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR, 
        3 * sizeof(float),
	0
    );

    ocl_buf_ul_room = opencl_buf_create (
        &ocl_dev, 
        CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR, 
        3 * sizeof(float),
	0
    );

    ocl_buf_incr_r = opencl_buf_create (
        &ocl_dev, 
        CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR, 
        3 * sizeof(float),
	0
    );

    ocl_buf_incr_c = opencl_buf_create (
        &ocl_dev, 
        CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR, 
        3 * sizeof(float),
	0
    );

    ocl_buf_nrm = opencl_buf_create (
        &ocl_dev, 
        CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR, 
        3 * sizeof(float),
	0
    );

    ocl_buf_lower_limit = opencl_buf_create (
        &ocl_dev, 
        CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR, 
        3 * sizeof(float),
	0
    );

    ocl_buf_upper_limit = opencl_buf_create (
        &ocl_dev, 
        CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR, 
        3 * sizeof(float),
	0
    );

    /* Copy ic to device (convert from double to float) */
    tmp[0] = proj->pmat->ic[0];
    tmp[1] = proj->pmat->ic[1];
    opencl_buf_write (&ocl_dev, ocl_buf_ic, 2 * sizeof(float), tmp);

    /* Copy p1 to device (convert from double to float) */
    tmp[0] = p1[0];
    tmp[1] = p1[1];
    tmp[2] = p1[2];
    opencl_buf_write (&ocl_dev, ocl_buf_p1, 3 * sizeof(float), tmp);

    /* Copy ul_room to device (convert from double to float) */
    tmp[0] = ul_room[0];
    tmp[1] = ul_room[1];
    tmp[2] = ul_room[2];
    opencl_buf_write (&ocl_dev, ocl_buf_ul_room, 3 * sizeof(float), tmp);

    /* Copy incr_r to device (convert from double to float) */
    tmp[0] = incr_r[0];
    tmp[1] = incr_r[1];
    tmp[2] = incr_r[2];
    opencl_buf_write (&ocl_dev, ocl_buf_incr_r, 3 * sizeof(float), tmp);

    /* Copy incr_c to device (convert from double to float) */
    tmp[0] = incr_c[0];
    tmp[1] = incr_c[1];
    tmp[2] = incr_c[2];
    opencl_buf_write (&ocl_dev, ocl_buf_incr_c, 3 * sizeof(float), tmp);

    /* Copy nrm to device (convert from double to float) */
    tmp[0] = pmat->nrm[0];
    tmp[1] = pmat->nrm[1];
    tmp[2] = pmat->nrm[2];
    opencl_buf_write (&ocl_dev, ocl_buf_nrm, 3 * sizeof(float), tmp);

    /* Copy lower_limit to device (convert from double to float) */
    tmp[0] = vol_limit->lower_limit[0];
    tmp[1] = vol_limit->lower_limit[1];
    tmp[2] = vol_limit->lower_limit[2];
    opencl_buf_write (&ocl_dev, ocl_buf_lower_limit, 3 * sizeof(float), tmp);
    
    /* Copy upper_limit to device (convert from double to float) */
    tmp[0] = vol_limit->upper_limit[0];
    tmp[1] = vol_limit->upper_limit[1];
    tmp[2] = vol_limit->upper_limit[2];
    opencl_buf_write (&ocl_dev, ocl_buf_upper_limit, 3 * sizeof(float), tmp);

    /* Convert sad from double to float */
    sad = proj->pmat->sad;

    /* Set drr kernel arguments */
    opencl_set_kernel_args (
	&ocl_dev, 
	sizeof (cl_mem), &ocl_buf_img[0], 
	sizeof (cl_mem), &ocl_buf_vol[0], 
	sizeof (cl_mem), &ocl_buf_vol_dim[0], 
#if defined (commentout)
	sizeof (cl_mem), &ocl_buf_vol_offset[0], 
	sizeof (cl_mem), &ocl_buf_vol_spacing[0], 
#endif
	sizeof (cl_mem), &ocl_buf_img_dim[0], 
	sizeof (cl_mem), &ocl_buf_ic[0], 
	sizeof (cl_mem), &ocl_buf_img_window[0], 
	sizeof (cl_mem), &ocl_buf_p1[0], 
	sizeof (cl_mem), &ocl_buf_ul_room[0], 
	sizeof (cl_mem), &ocl_buf_incr_r[0], 
	sizeof (cl_mem), &ocl_buf_incr_c[0], 
	sizeof (cl_mem), &ocl_buf_nrm[0], 
	sizeof (cl_mem), &ocl_buf_lower_limit[0], 
	sizeof (cl_mem), &ocl_buf_upper_limit[0], 
	sizeof (cl_float), &sad, 
	sizeof (cl_float), &options->scale,
	(size_t) 0
    );

    /* Compute workgroup size */
    /* (Max local_work_size for my ATI RV710 is 128) */
    size_t local_work_size = 128;
    size_t global_work_size = (float) proj->dim[0] * proj->dim[1];

    /* Invoke kernel */
    opencl_kernel_enqueue (&ocl_dev, global_work_size, local_work_size);

    /* Read back results */
    opencl_buf_read (&ocl_dev, ocl_buf_img, 
	sizeof (float) * proj->dim[0] * proj->dim[1], proj->img);
}
