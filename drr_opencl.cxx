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

    /* Set up devices and kernels */
    opencl_open_device (&ocl_dev);
    opencl_load_programs (&ocl_dev, "drr_opencl.cl");
    opencl_kernel_create (&ocl_dev, "kernel_drr");
}
