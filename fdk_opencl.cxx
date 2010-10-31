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
    cl_uint buf_entries;
    cl_uint buf_size;
    cl_uint *buf_in, *buf_out;
    cl_uint multiplier = 2;
    Proj_image *cbi;
    int image_num;

    /* Set up devices and kernels */
    opencl_open_device (&ocl_dev);
    opencl_load_programs (&ocl_dev, "fdk_opencl.cl");
    opencl_kernel_create (&ocl_dev, "kernel_2");

    /* Retrieve 2D image to get dimensions */
    cbi = proj_image_dir_load_image (proj_dir, 0);

    /* Set up device memory */
    ocl_buf_vol = opencl_buf_create (
	&ocl_dev, vol->pix_size * vol->npix, vol->img);
    ocl_buf_img = opencl_buf_create (
	&ocl_dev, cbi->dim[1] * cbi->dim[0] * sizeof(float), 0);

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

	/* Set fdk kernel arguments */
	opencl_set_kernel_args (
	    &ocl_dev, 
	    sizeof (cl_mem), 
	    &ocl_buf_vol[0], 
	    sizeof (cl_mem), 
	    &ocl_buf_img[0], 
	    //	    sizeof (cl_uint), 
	    //	    &multiplier, 
	    (size_t) 0
	);

	/* Invoke kernel */
	size_t global_work_size = 100;
	size_t local_work_size = 100;
	opencl_kernel_enqueue (&ocl_dev, global_work_size, local_work_size);
    }

#if defined (commentout)
    /* Read back results */
    opencl_buf_read (&ocl_dev, ocl_buf_out, buf_size, buf_out);

    for (cl_uint i = 0; i < buf_entries; i++) {
	printf ("%d ", buf_out[i]);
    }
    printf ("\n");
#endif
}
