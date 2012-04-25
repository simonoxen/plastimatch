/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _fdk_utils_h_
#define _fdk_utils_h_

#include "plm_config.h"
#include "plmbase.h"
#include "fdk_opts.h"
#include "proj_image.h"

#if defined __cplusplus
extern "C" {
#endif

void convert_to_hu (Volume* vol, Fdk_options* options);
Volume* my_create_volume (Fdk_options* options);
Proj_image* get_image_pfm (Fdk_options* options, int image_num);
Proj_image* get_image_raw (Fdk_options* options, int image_num);
void write_coronal_sagittal (Fdk_options* options, Volume* vol);


#if defined __cplusplus
}
#endif

#endif
