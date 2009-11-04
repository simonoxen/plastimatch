/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _proj_image_h_
#define _proj_image_h_

#include "volume.h"
#include "fdk.h"
#include "fdk_opts.h"

#if defined __cplusplus
extern "C" {
#endif

CB_Image* get_image (Fdk_options* options, int image_num);
CB_Image* proj_image_load_pfm (char* img_filename, char* mat_filename);
void free_cb_image (CB_Image* cbi);


#if defined __cplusplus
}
#endif

#endif
