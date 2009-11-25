/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _proj_image_h_
#define _proj_image_h_

#include "plm_config.h"
#include "volume.h"
#include "fdk.h"
#include "fdk_opts.h"

#if defined __cplusplus
extern "C" {
#endif

gpuit_EXPORT
CB_Image* get_image (Fdk_options* options, int image_num);
gpuit_EXPORT
CB_Image* proj_image_load_pfm (char* img_filename, char* mat_filename);
gpuit_EXPORT
CB_Image* 
proj_image_load_and_filter (
    Fdk_options * options, 
    char* img_filename, 
    char* mat_filename
);

gpuit_EXPORT
void free_cb_image (CB_Image* cbi);

#if defined __cplusplus
}
#endif

#endif
