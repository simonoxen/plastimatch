/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "plm_config.h"
#include "fdk.h"
#include "fdk_opts.h"
#include "fdk_utils.h"
#include "proj_image.h"
#include "ramp_filter.h"
#include "volume.h"

Volume*
my_create_volume (Fdk_options* options)
{
    float offset[3];
    float spacing[3];
    float* vol_size = options->vol_size;
    int* resolution = options->resolution;

    spacing[0] = vol_size[0] / resolution[0];
    spacing[1] = vol_size[1] / resolution[1];
    spacing[2] = vol_size[2] / resolution[2];

    offset[0] = -vol_size[0] / 2.0f + spacing[0] / 2.0f;
    offset[1] = -vol_size[1] / 2.0f + spacing[1] / 2.0f;
    offset[2] = -vol_size[2] / 2.0f + spacing[2] / 2.0f;

    return volume_create (resolution, offset, spacing, PT_FLOAT, 0, 0);
}

float
convert_to_hu_pixel (float in_value)
{
    float hu;
    float diameter = 40.0;  /* reconstruction diameter in cm */
    hu = 1000 * ((in_value / diameter) - .167) / .167;
    return hu;
}

void
convert_to_hu (Volume* vol, Fdk_options* options)
{
    int i, j, k, p;
    float* img = (float*) vol->img;
    
    p = 0;
    for (k = 0; k < vol->dim[2]; k++) {
	for (j = 0; j < vol->dim[1]; j++) {
	    for (i = 0; i < vol->dim[0]; i++) {
		img[p] = convert_to_hu_pixel (img[p]);
		p++;
	    }
	}
    }
}

CB_Image*
get_image (Fdk_options* options, int image_num)
{
#if defined (READ_PFM)
    char* img_file_pat = "out_%04d.pfm";
#else
    char* img_file_pat = "out_%04d.pgm";
#endif
    char* mat_file_pat = "out_%04d.txt";

    char img_file[1024], mat_file[1024], fmt[1024];
    sprintf (fmt, "%s/%s", options->input_dir, img_file_pat);
    sprintf (img_file, fmt, image_num);
    sprintf (fmt, "%s/%s", options->input_dir, mat_file_pat);
    sprintf (mat_file, fmt, image_num);
    return proj_image_load_pfm (img_file, mat_file);
}

#if defined (commentout)
void
free_cb_image (CB_Image* cbi)
{
    free (cbi->img);
    free (cbi);
}
#endif
