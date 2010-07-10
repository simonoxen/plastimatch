/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plm_config.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "demons_opts.h"
#include "demons_misc.h"
#include "mha_io.h"
#include "opencl_utils.h"
#include "plm_timer.h"
#include "print_and_exit.h"
#include "volume.h"

Volume*
demons_opencl (
    Volume* fixed, 
    Volume* moving, 
    Volume* moving_grad, 
    Volume* vf_init, 
    DEMONS_Parms* parms)
{
    Opencl_device ocl_dev;
    opencl_open_device (&ocl_dev);
    opencl_close_device (&ocl_dev);
    exit (0);
    return 0;
}
