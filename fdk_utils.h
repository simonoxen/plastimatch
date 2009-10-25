/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _fdk_utils_h_
#define _fdk_utils_h_

#include "volume.h"
#include "fdk.h"
#include "fdk_opts.h"

#if defined __cplusplus
extern "C" {
#endif

void convert_to_hu (Volume* vol, MGHCBCT_Options* options);
Volume* my_create_volume (MGHCBCT_Options* options);
CB_Image* get_image (MGHCBCT_Options* options, int image_num);
CB_Image* load_cb_image (char* img_filename, char* mat_filename);
void free_cb_image (CB_Image* cbi);


#if defined __cplusplus
}
#endif

#endif
