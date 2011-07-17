/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _rasterize_slice_h_
#define _rasterize_slice_h_

#include "plm_config.h"

#if defined __cplusplus
extern "C" {
#endif

plastimatch1_EXPORT
void
rasterize_slice (
    unsigned char* acc_img,
    int* dims,
    float* spacing,
    float* offset,
    int num_vertices,
    float* x_in,           /* vertices in mm */
    float* y_in            /* vertices in mm */
);

#if defined __cplusplus
}
#endif

#endif
