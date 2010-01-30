/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _drr_h_
#define _drr_h_

#include "drr_opts.h"
#include "math_util.h"
#include "proj_image.h"
#include "volume.h"

#define DRR_PLANE_RAY_TOLERANCE 1e-8
#define DRR_STRIDE_TOLERANCE 1e-10
#define DRR_HUGE_DOUBLE 1e10
#define DRR_LEN_TOLERANCE 1e-6
#define DRR_TOPLANE_TOLERANCE 1e-7
#define DRR_BOUNDARY_TOLERANCE 1e-6

#define DRR_MSD_NUM_BINS 60

#define DRR_PREPROCESS_ATTENUATION 1


#if defined __cplusplus
extern "C" {
#endif
gpuit_EXPORT
void
drr_render_volume_perspective (
    Proj_image *proj,
    Volume *vol, 
    double ps[2], 
    char *multispectral_fn, 
    Drr_options *options
);
gpuit_EXPORT
void
drr_preprocess_attenuation (Volume* vol);

#if defined __cplusplus
}
#endif

#endif
