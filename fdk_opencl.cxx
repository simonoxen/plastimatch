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
    Opencl_buf *ocl_buf_in, *ocl_buf_out;
    cl_uint buf_entries;
    cl_uint buf_size;
    cl_uint *buf_in, *buf_out;
    cl_uint multiplier = 2;

    opencl_open_device (&ocl_dev);
    opencl_load_programs (&ocl_dev, "fdk_opencl.cl");

    buf_entries = 100;
    buf_size = buf_entries * sizeof (cl_uint);
    buf_in = (cl_uint*) malloc (buf_size);
    buf_out = (cl_uint*) malloc (buf_size);
    for (cl_uint i = 0; i < buf_entries; i++) {
	buf_in[i] = i;
	buf_out[i] = 0;
    }

    ocl_buf_in = opencl_buf_create (&ocl_dev, buf_size, buf_in);
    ocl_buf_out = opencl_buf_create (&ocl_dev, buf_size, buf_out);
    opencl_kernel_create (&ocl_dev, "kernel_1");

    opencl_set_kernel_args (
	&ocl_dev, 
	sizeof (cl_mem), 
	&ocl_buf_out[0], 
	sizeof (cl_mem), 
	&ocl_buf_in[0], 
	sizeof (cl_uint), 
	&multiplier, 
	0
    );
    opencl_kernel_enqueue (&ocl_dev, buf_entries, 1);


}
