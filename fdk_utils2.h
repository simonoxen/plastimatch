/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _fdk_utils2_h_
#define _fdk_utils2_h_

#include "volume.h"
#include "fdk_opts_ext.h"
#include "fdk.h"
#if defined __cplusplus
extern "C" {
#endif
void convert_to_hu (Volume* vol, MGHCBCT_Options_ext* options);
Volume* my_create_volume (MGHCBCT_Options_ext* options);
CB_Image* get_image (MGHCBCT_Options_ext* options, int image_num);
CB_Image* load_cb_image (char* img_filename, char* mat_filename);
void free_cb_image (CB_Image* cbi);
void bowtie_correction(Volume * vol,MGHCBCT_Options_ext* options);
#if defined __cplusplus
}
#endif

#endif
