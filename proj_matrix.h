/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _proj_matrix_h_
#define _proj_matrix_h_

typedef struct proj_matrix Proj_matrix;
struct proj_matrix
{
    double ic[2];	  /* Image Center:  ic[0] = x, ic[1] = y */
    double matrix[12];	  /* Projection matrix */
    double sad;		  /* Distance: Source To Axis */
    double sid;		  /* Distance: Source to Image */
    double cam[3];	  /* Location of camera */
    double nrm[3];	  /* Ray from image center to source */

    double extrinsic[16];
    double intrinsic[12];
};


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
proj_matrix_destroy (Proj_matrix* pmat);

gpuit_EXPORT
void
proj_matrix_set (
    Proj_matrix *pmat,
    double* cam, 
    double* tgt, 
    double* vup, 
    double sid, 
    double* ic, 
    double* ps, 
    int* ires
);

gpuit_EXPORT
void
proj_matrix_debug (
    Proj_matrix *pmat
);

gpuit_EXPORT
void
proj_matrix_get_nrm (
    Proj_matrix *pmat,
    double nrm[3]
);

gpuit_EXPORT
void
proj_matrix_get_pdn (
    Proj_matrix *pmat,
    double pdn[3]
);

gpuit_EXPORT
void
proj_matrix_get_prt (
    Proj_matrix *pmat,
    double prt[3]
);

gpuit_EXPORT
void
proj_matrix_save (
    Proj_matrix *pmat,
    const char *fn
);

#ifdef __cplusplus
}
#endif

#endif
