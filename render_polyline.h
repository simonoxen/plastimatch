/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _render_polyline_h_
#define _render_polyline_h_

#include "plm_config.h"

#if defined __cplusplus
extern "C" {
#endif

plastimatch1_EXPORT
void
render_slice_polyline (unsigned char* acc_img,
		    int* dims,
		    float* spacing,
		    float* offset, 
		    int num_vertices,
		    float* x,
		    float* y);

#if defined __cplusplus
}
#endif

#endif
