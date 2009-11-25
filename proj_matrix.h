/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _proj_matrix_h_
#define _proj_matrix_h_

#include "MGHMtx_opts.h"
#ifdef __cplusplus
extern "C" {
#endif

gpuit_EXPORT
void
proj_matrix_write (double* cam, 
		   double* tgt, double* vup,
		   double sid, double* ic,
		   double* ps, int* ires,
		   int varian_mode, 
		   char* out_fn);
void write_matrix (MGHMtx_Options* options);
void wm_set_default_options (MGHMtx_Options* options);

#ifdef __cplusplus
}
#endif

#endif
