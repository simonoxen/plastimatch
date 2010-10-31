/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plm_config.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "autotune_opencl.h"
#include "fdk_opencl_p.h"
#include "fdk_opts.h"
#include "fdk_util.h"
#include "math_util.h"
#include "mha_io.h"
#include "opencl_util.h"
#include "opencl_util_nvidia.h"
#include "plm_timer.h"
#include "proj_image.h"
#include "proj_image_dir.h"
#include "volume.h"

void 
opencl_reconstruct_conebeam (
    Volume *vol, 
    Proj_image_dir *proj_dir, 
    Fdk_options *options
)
{
    Opencl_device ocl_dev;
    Opencl_buf *ocl_buf_vol;
    Opencl_buf *ocl_buf_img;
    Opencl_buf *ocl_buf_matrix;
    Opencl_buf *ocl_buf_vol_dim;
    cl_uint multiplier = 2;
    float *img = (float*) vol->img;
    Proj_image *cbi;
    int image_num;

    /* Set up devices and kernels */
    opencl_open_device (&ocl_dev);
    opencl_load_programs (&ocl_dev, "fdk_opencl.cl");
    opencl_kernel_create (&ocl_dev, "kernel_2");

    /* Retrieve 2D image to get dimensions */
    cbi = proj_image_dir_load_image (proj_dir, 0);

    /* Set up device memory */
    ocl_buf_vol = opencl_buf_create (&ocl_dev, 
	CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, 
	vol->pix_size * vol->npix, vol->img);
    ocl_buf_img = opencl_buf_create (&ocl_dev, 
	CL_MEM_READ_ONLY, 
	cbi->dim[1] * cbi->dim[0] * sizeof(float), 0);
    ocl_buf_matrix = opencl_buf_create (&ocl_dev, 
	CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR, 
	12 * sizeof(float), 0);
    ocl_buf_vol_dim = opencl_buf_create (&ocl_dev, 
	CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, 
	3 * sizeof(int), vol->dim);

    /* Free cbi image */
    proj_image_destroy (cbi);

    /* Project each image into the volume one at a time */
    for (image_num = options->first_img; 
	 image_num < proj_dir->num_proj_images; 
	 image_num++)
    {
	/* Load the current image and properties */
	cbi = proj_image_dir_load_image(proj_dir, image_num);

	/* Copy image bytes to device */
	opencl_buf_write (&ocl_dev, ocl_buf_img, 
	    cbi->dim[1] * cbi->dim[0] * sizeof(float), cbi->img);

	/* Copy matrix to device */
	opencl_buf_write (&ocl_dev, ocl_buf_matrix, 
	    12 * sizeof(float), cbi->pmat->matrix);

	/* Set fdk kernel arguments */
	opencl_set_kernel_args (
	    &ocl_dev, 
	    sizeof (cl_mem), 
	    &ocl_buf_vol[0], 
	    sizeof (cl_mem), 
	    &ocl_buf_img[0], 
	    sizeof (cl_mem), 
	    &ocl_buf_matrix[0], 
	    sizeof (cl_mem), 
	    &ocl_buf_vol_dim[0], 
	    //	    sizeof (cl_uint), 
	    //	    &multiplier, 
	    (size_t) 0
	);

	/* Compute workgroup size */
	//size_t local_work_size = 512;
	size_t local_work_size = 128;
	size_t global_work_size = (float) vol->npix;

#if defined (commentout)
	local_work_size = 4;
	global_work_size = 8;
#endif

	/* Invoke kernel */
	opencl_kernel_enqueue (&ocl_dev, global_work_size, local_work_size);
    }

    /* Read back results */
    opencl_buf_read (&ocl_dev, ocl_buf_vol, 
	vol->pix_size * vol->npix, vol->img);

    int num_nonzero = 0;
    for (cl_uint i = 0; i < vol->npix; i++) {
	if (img[i] != 0.0f) {
	    printf ("[%d] %f\n", i, img[i]);
	    if (++num_nonzero == 10) {
		break;
	    }
	}
    }
}
