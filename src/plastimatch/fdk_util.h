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

void convert_to_hu (Volume* vol, Fdk_parms* parms);
Volume* my_create_volume (Fdk_parms* parms);
Proj_image* get_image_pfm (Fdk_parms* parms, int image_num);
Proj_image* get_image_raw (Fdk_parms* parms, int image_num);
void write_coronal_sagittal (Fdk_parms* parms, Volume* vol);


#if defined __cplusplus
}
#endif

#endif
