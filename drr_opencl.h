#ifndef _DRR_OPENCL_H_
#define _DRR_OPENCL_H_

/*****************
*  C   #includes *
*****************/
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "volume.h"
#include "drr_opts.h"

#if defined __cplusplus
extern "C" {
#endif

void drr_opencl_render_volume (Volume* vol, Drr_options* options);

#if defined __cplusplus
}
#endif

#endif