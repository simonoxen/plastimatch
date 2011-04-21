/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plm_config.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "file_util.h"
#include "math_util.h"
#include "proj_matrix.h"
#include "volume.h"

/* -----------------------------------------------------------------------
   Private functions
   ----------------------------------------------------------------------- */
/* -----------------------------------------------------------------------
   Public functions
   ----------------------------------------------------------------------- */
void
proj_matrix_init (Proj_matrix *pmat)
{
    memset (pmat, 0, sizeof(Proj_matrix));
}

Proj_matrix*
proj_matrix_create (void)
{
    Proj_matrix *pmat;
    
    pmat = (Proj_matrix*) malloc (sizeof(Proj_matrix));
    if (!pmat) return 0;

    proj_matrix_init (pmat);

    return pmat;
}

Proj_matrix*
proj_matrix_clone (Proj_matrix* pmat_in)
{
    Proj_matrix *pmat;
    
    pmat = proj_matrix_create ();
    if (!pmat) return 0;

    /* No dynamically allocated memory in proj_matrix */
    memcpy (pmat, pmat_in, sizeof (Proj_matrix));

    return pmat;
}

void
proj_matrix_destroy (Proj_matrix* pmat)
{
    free (pmat);
}

static
void
proj_matrix_write (
    Proj_matrix *pmat,
    FILE *fp
)
{
    fprintf (fp, "%18.8e %18.8e\n", pmat->ic[0], pmat->ic[1]);
    fprintf (fp,
	"%18.8e %18.8e %18.8e %18.8e\n" 
	"%18.8e %18.8e %18.8e %18.8e\n" 
	"%18.8e %18.8e %18.8e %18.8e\n", 
	pmat->matrix[0], pmat->matrix[1], pmat->matrix[2], pmat->matrix[3],
	pmat->matrix[4], pmat->matrix[5], pmat->matrix[6], pmat->matrix[7],
	pmat->matrix[8], pmat->matrix[9], pmat->matrix[10], pmat->matrix[11]
    );
    fprintf (fp, "%18.8e\n%18.8e\n", pmat->sad, pmat->sid);

    /* NRM */
    //fprintf (fp, "%18.8e %18.8e %18.8e\n", nrm[0], nrm[1], nrm[2]);
    fprintf (fp, "%18.8e %18.8e %18.8e\n", pmat->extrinsic[8], 
	pmat->extrinsic[9], pmat->extrinsic[10]);

    fprintf (fp,
	"Extrinsic\n"
	"%18.8e %18.8e %18.8e %18.8e\n" 
	"%18.8e %18.8e %18.8e %18.8e\n" 
	"%18.8e %18.8e %18.8e %18.8e\n" 
	"%18.8e %18.8e %18.8e %18.8e\n", 
	pmat->extrinsic[0], pmat->extrinsic[1], pmat->extrinsic[2], 
	pmat->extrinsic[3], pmat->extrinsic[4], pmat->extrinsic[5], 
	pmat->extrinsic[6], pmat->extrinsic[7], pmat->extrinsic[8], 
	pmat->extrinsic[9], pmat->extrinsic[10], pmat->extrinsic[11],
	pmat->extrinsic[12], pmat->extrinsic[13], pmat->extrinsic[14], 
	pmat->extrinsic[15]
    );
    fprintf (fp,
	"Intrinsic\n"
	"%18.8e %18.8e %18.8e %18.8e\n" 
	"%18.8e %18.8e %18.8e %18.8e\n" 
	"%18.8e %18.8e %18.8e %18.8e\n", 
	pmat->intrinsic[0], pmat->intrinsic[1], 
	pmat->intrinsic[2], pmat->intrinsic[3],
	pmat->intrinsic[4], pmat->intrinsic[5], 
	pmat->intrinsic[6], pmat->intrinsic[7],
	pmat->intrinsic[8], pmat->intrinsic[9], 
	pmat->intrinsic[10], pmat->intrinsic[11]
    );
}

void
proj_matrix_debug (
    Proj_matrix *pmat
)
{
    proj_matrix_write (pmat, stdout);
}

void
proj_matrix_save (
    Proj_matrix *pmat,
    const char *fn
)
{
    FILE *fp;

    if (!fn) return;
    if (!pmat) return;

    make_directory_recursive (fn);
    fp = fopen (fn, "w");
    if (!fp) {
	fprintf (stderr, "Error opening %s for write\n", fn);
	exit (-1);
    }

    proj_matrix_write (pmat, fp);

    fclose (fp);
}

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
)
{
    const int cols = 4;
    double nrm[3];       /* Panel normal */
    double plt[3];       /* Panel left (toward first column) */
    double pup[3];       /* Panel up (toward top row) */

    vec3_copy (pmat->cam, cam);
    pmat->sid = sid;
    pmat->sad = vec3_len (cam);
    pmat->ic[0] = ic[0];
    pmat->ic[1] = ic[1];

    /* Compute imager coordinate sys (nrm,pup,plt) 
       ---------------
       nrm = cam - tgt
       plt = nrm x vup
       pup = plt x nrm
       ---------------
    */
    vec3_sub3 (nrm, cam, tgt);
    vec3_normalize1 (nrm);
    vec3_cross (plt, nrm, vup);
    vec3_normalize1 (plt);
    vec3_cross (pup, plt, nrm);
    vec3_normalize1 (pup);

#if defined (commentout)
    printf ("CAM = %g %g %g\n", cam[0], cam[1], cam[2]);
    printf ("TGT = %g %g %g\n", tgt[0], tgt[1], tgt[2]);
    printf ("NRM = %g %g %g\n", nrm[0], nrm[1], nrm[2]);
    printf ("PLT = %g %g %g\n", plt[0], plt[1], plt[2]);
    printf ("PUP = %g %g %g\n", pup[0], pup[1], pup[2]);
#endif

    /* Build extrinsic matrix - rotation part */
    vec_zero (pmat->extrinsic, 16);
    vec3_copy (&pmat->extrinsic[0], plt);
    vec3_copy (&pmat->extrinsic[4], pup);
    vec3_copy (&pmat->extrinsic[8], nrm);
    vec3_invert (&pmat->extrinsic[0]);
    vec3_invert (&pmat->extrinsic[4]);
    vec3_invert (&pmat->extrinsic[8]);
    m_idx (pmat->extrinsic,cols,3,3) = 1.0;

    /* Build extrinsic matrix - translation part */
    pmat->extrinsic[3] = vec3_dot (plt, tgt);
    pmat->extrinsic[7] = vec3_dot (pup, tgt);
    pmat->extrinsic[11] = vec3_dot (nrm, tgt) + pmat->sad;

#if defined (commentout)
    printf ("EXTRINSIC\n%g %g %g %g\n%g %g %g %g\n"
	"%g %g %g %g\n%g %g %g %g\n",
	pmat->extrinsic[0], pmat->extrinsic[1], 
	pmat->extrinsic[2], pmat->extrinsic[3],
	pmat->extrinsic[4], pmat->extrinsic[5], 
	pmat->extrinsic[6], pmat->extrinsic[7],
	pmat->extrinsic[8], pmat->extrinsic[9], 
	pmat->extrinsic[10], pmat->extrinsic[11],
	pmat->extrinsic[12], pmat->extrinsic[13], 
	pmat->extrinsic[14], pmat->extrinsic[15]);
#endif

    /* Build intrinsic matrix */
    vec_zero (pmat->intrinsic, 12);
    m_idx (pmat->intrinsic,cols,0,0) = 1 / ps[0];
    m_idx (pmat->intrinsic,cols,1,1) = 1 / ps[1];
    m_idx (pmat->intrinsic,cols,2,2) = 1 / sid;

#if defined (commentout)
    printf ("INTRINSIC\n%g %g %g %g\n%g %g %g %g\n%g %g %g %g\n",
	pmat->intrinsic[0], pmat->intrinsic[1], 
	pmat->intrinsic[2], pmat->intrinsic[3],
	pmat->intrinsic[4], pmat->intrinsic[5], 
	pmat->intrinsic[6], pmat->intrinsic[7],
	pmat->intrinsic[8], pmat->intrinsic[9], 
	pmat->intrinsic[10], pmat->intrinsic[11]);
#endif

    /* Build projection matrix */
    mat_mult_mat (pmat->matrix, pmat->intrinsic,3,4, pmat->extrinsic,4,4);
}

void
proj_matrix_get_nrm (
    Proj_matrix *pmat,
    double nrm[3]
)
{
    vec3_copy (nrm, &pmat->extrinsic[8]);
    vec3_invert (nrm);
}

void
proj_matrix_get_pdn (
    Proj_matrix *pmat,
    double pdn[3]
)
{
    vec3_copy (pdn, &pmat->extrinsic[4]);
}

void
proj_matrix_get_prt (
    Proj_matrix *pmat,
    double prt[3]
)
{
    vec3_copy (prt, &pmat->extrinsic[0]);
}
