/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _fdk_utils_h_ext
#define _fdk_utils_h_ext

#include "volume.h"
#include "fdk.h"
#include "fdk_opts_ext.h"

#if defined __cplusplus
extern "C" {
#endif

void convert_to_hu (Volume* vol, MGHCBCT_Options_ext * options);
Volume* my_create_volume (MGHCBCT_Options_ext * options);
CB_Image* get_image (MGHCBCT_Options_ext * options, int image_num);
int write_image (CB_Image* cbi, MGHCBCT_Options_ext* options, int image_num);
CB_Image* load_cb_image (char* img_filename, char* mat_filename);
CB_Image* load_and_filter_cb_image (MGHCBCT_Options_ext * options, char* img_filename, char* mat_filename);
void free_cb_image (CB_Image* cbi);


void bowtie_correction(Volume * vol,MGHCBCT_Options_ext* options);

#if defined __cplusplus
}
#endif

#endif
