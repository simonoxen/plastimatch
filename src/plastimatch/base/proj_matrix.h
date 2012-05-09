/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _proj_matrix_h_
#define _proj_matrix_h_

#include "plmbase_config.h"

class Proj_matrix {
public:
    Proj_matrix ();

public:
    double ic[2];	  /* Image Center:  ic[0] = x, ic[1] = y */
    double matrix[12];	  /* Projection matrix */
    double sad;		  /* Distance: Source To Axis */
    double sid;		  /* Distance: Source to Image */
    double cam[3];	  /* Location of camera */
    double nrm[3];	  /* Ray from image center to source */

    double extrinsic[16];
    double intrinsic[12];
};

PLMBASE_C_API Proj_matrix* proj_matrix_clone (Proj_matrix* pmat_in);
PLMBASE_C_API void proj_matrix_set (
    Proj_matrix *pmat,
    double* cam, 
    double* tgt, 
    double* vup, 
    double sid, 
    double* ic, 
    double* ps, 
    int* ires
);
PLMBASE_C_API void proj_matrix_debug (Proj_matrix *pmat);
PLMBASE_C_API void proj_matrix_get_nrm (Proj_matrix *pmat, double nrm[3]);
PLMBASE_C_API void proj_matrix_get_pdn (Proj_matrix *pmat, double pdn[3]);
PLMBASE_C_API void proj_matrix_get_prt (Proj_matrix *pmat, double prt[3]);
PLMBASE_C_API void proj_matrix_save (Proj_matrix *pmat, const char *fn);

#endif
