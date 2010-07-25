/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _landmark_warp_opts_h_
#define _landmark_warp_opts_h_

#include "plm_config.h"
#include "tps.h"

typedef struct landmark_warp_options Landmark_warp_options;
struct landmark_warp_options {
    char *input_moving_landmarks_fn;
    char *input_fixed_landmarks_fn;
    char *input_vf_fn;
    char *input_xform_fn;
    char *input_moving_image_fn;
    char *output_warped_image_fn;
    char *output_vf_fn;
    float *rbf_radius;
};

void landmark_warp_opts_parse_args (Landmark_warp_options* options, int argc, char* argv[]);

#endif
