/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _proj_matrix_h_
#define _proj_matrix_h_

typedef struct proj_matrix Proj_matrix;
struct proj_matrix
{
    double ic[2];	/* Image Center:  ic[0] = x, ic[1] = y     */
    double matrix[12];	/* Projection matrix */
    double sad;		/* Distance: Source To Axis */
    double sid;		/* Distance: Source to Image */
    double nrm[3];	/* Ray from image center to source */
};


#include "MGHMtx_opts.h"
#ifdef __cplusplus
extern "C" {
#endif

gpuit_EXPORT 
Proj_matrix*
proj_matrix_create ();

gpuit_EXPORT 
void
proj_matrix_init (Proj_matrix* matrix);

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
