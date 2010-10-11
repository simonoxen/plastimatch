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
    opencl_open_device (&ocl_dev);
    opencl_load_programs (&ocl_dev, "fdk_opencl.cl");

}
