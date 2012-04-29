/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _rasterize_slice_h_
#define _rasterize_slice_h_

#include "plm_config.h"

#include "plmsys.h"

#if defined __cplusplus
extern "C" {
#endif

plastimatch1_EXPORT
void
rasterize_slice (
    unsigned char* acc_img,
    plm_long* dims,
    float* spacing,
    float* offset,
    size_t num_vertices,
    const float* x_in,           /* polygon vertices in mm */
    const float* y_in            /* polygon vertices in mm */
);

bool
point_in_polygon (
    const float* x_in,           /* polygon vertices in mm */
    const float* y_in,           /* polygon vertices in mm */
    size_t num_vertices,
    float x_test,
    float y_test
);

#if defined __cplusplus
}
#endif

#endif
