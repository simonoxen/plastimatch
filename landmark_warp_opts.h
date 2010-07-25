/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _landmark_warp_opts_h_
#define _landmark_warp_opts_h_

#include "plm_config.h"
#include "tps.h"

typedef struct tps_options Tps_options;
struct tps_options {
    char *tps_xf_fn;
    char *moving_image_fn;
    char *output_warped_fn;
    char *output_vf_fn;
    char *moving_landmarks_fn;
    char *fixed_landmarks_fn;
};

void tps_warp_opts_parse_args (Tps_options* options, int argc, char* argv[]);

#endif
