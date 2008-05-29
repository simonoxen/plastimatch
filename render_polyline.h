/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _render_polyline_h_
#define _render_polyline_h_

void
render_slice_polyline (unsigned char* acc_img,
		    int* dims,
		    float* spacing,
		    float* offset, 
		    int num_vertices,
		    float* x,
		    float* y);


#endif
