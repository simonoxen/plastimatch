/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _drr_h_
#define _drr_h_

#include "plm_config.h"
#include "plmbase.h"
#include "drr_opts.h"
#include "plm_math.h"
#include "proj_image.h"

#define DRR_PLANE_RAY_TOLERANCE 1e-8
#define DRR_STRIDE_TOLERANCE 1e-10
#define DRR_HUGE_DOUBLE 1e10
#define DRR_LEN_TOLERANCE 1e-6
#define DRR_TOPLANE_TOLERANCE 1e-7
#define DRR_BOUNDARY_TOLERANCE 1e-6

#define DRR_MSD_NUM_BINS 60

#define DRR_PREPROCESS_ATTENUATION 1

//#define DRR_VERBOSE 1
//#define DRR_DEBUG_CALLBACK 1
//#define DRR_ULTRA_VERBOSE 1

#if defined __cplusplus
extern "C" {
#endif

gpuit_EXPORT
void
drr_render_volume_perspective (
    Proj_image *proj,
    Volume *vol, 
    double ps[2], 
    void *dev_state, 
    Drr_options *options
);
gpuit_EXPORT
void
drr_preprocess_attenuation (Volume* vol);
gpuit_EXPORT
void preprocess_attenuation_and_drr_render_volume_cl (Volume* vol, Drr_options* options);

#if defined __cplusplus
}
#endif

#endif
