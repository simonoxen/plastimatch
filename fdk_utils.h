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

void convert_to_hu (Volume* vol, Fdk_options* options);
Volume* my_create_volume (Fdk_options* options);
CB_Image* get_image_pfm (Fdk_options* options, int image_num);
CB_Image* get_image_raw (Fdk_options* options, int image_num);
void write_coronal_sagittal (Fdk_options* options, Volume* vol);


#if defined __cplusplus
}
#endif

#endif
