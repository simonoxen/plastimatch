/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _proj_image_h_
#define _proj_image_h_

#include "plm_config.h"
#include "volume.h"
#include "fdk_opts.h"

typedef struct proj_image Proj_image;
struct proj_image
{
    int dim[2];         /* dim[0] = cols
			   dim[1] = rows */
    double ic[2];	/* Image Center
			   ic[0] = x
			   ic[1] = y     */
    double matrix[12];	// Projection matrix
    double sad;		// Distance: Source To Axis
    double sid;		// Distance: Source to Image
    double nrm[3];	// Ray from image center to source
    float* img;		// Pixel data
};

#if defined __cplusplus
extern "C" {
#endif

gpuit_EXPORT
Proj_image*
proj_image_create (void);

gpuit_EXPORT Proj_image* 
proj_image_load_and_filter (
    Fdk_options * options, 
    char* img_filename, 
    char* mat_filename
);

gpuit_EXPORT Proj_image* 
proj_image_load (
    char* img_filename,
    char* mat_filename
);

gpuit_EXPORT void
proj_image_debug_header (Proj_image *proj);

gpuit_EXPORT void
proj_image_stats (Proj_image *proj);

gpuit_EXPORT void 
proj_image_free (Proj_image* proj);

#if defined __cplusplus
}
#endif

#endif
