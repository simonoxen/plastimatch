/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include "plm_config.h"
#include "f2c.h"
#include "volume.h"
#include "bspline_opts.h"
#include "bspline.h"
#include "logfile.h"

extern void bspline_initialize_streams_on_gpu(Volume* fixed, Volume* moving, Volume* moving_grad, BSPLINE_Xform *bxf, BSPLINE_Parms *parms);


void
setulb_ (integer*       n,
	 integer*       m,
	 doublereal*    x,
	 doublereal*    l,
	 doublereal*    u,
	 integer*       nbd,
	 doublereal*    f,
	 doublereal*    g,
	 doublereal*    factr,
	 doublereal*    pgtol,
	 doublereal*    wa,
	 integer*       iwa,
	 char*          task,
	 integer*       iprint,
	 char*          csave,
	 logical*       lsave,
	 integer*       isave,
	 doublereal*    dsave,
	 ftnlen         task_len,
	 ftnlen         csave_len
	 );

#if defined (commentout)
#define N 25
#define M 5
#define NMAX 1024
#define MMAX 17
#define NMAX 1024
#define MMAX 17
//#define N 25
//#define M 5
#define N NMAX
#define M MMAX
#endif


#if defined (commentout)
SAVEME ()
{

    char task[60], csave[60];
    logical lsave[4];
#if defined (commentout)
    integer n, m, iprint, nbd[NMAX], iwa[3*NMAX], isave[44];
    doublereal f, factr, pgtol, x[NMAX], l[NMAX], u[NMAX], g[NMAX],
	    wa[2*MMAX*NMAX+4*NMAX+12*MMAX*MMAX+12*MMAX], dsave[29];
#endif

    doublereal t1, t2;
    integer i;


    nbd = (integer*) malloc (sizeof(integer)*NMAX);
    iwa = (integer*) malloc (sizeof(integer)*3*NMAX);
    x = (doublereal*) malloc (sizeof(doublereal)*NMAX);
    l = (doublereal*) malloc (sizeof(doublereal)*NMAX);
    u = (doublereal*) malloc (sizeof(doublereal)*NMAX);
    g = (doublereal*) malloc (sizeof(doublereal)*NMAX);
    wa = (doublereal*) malloc (sizeof(doublereal)*2*MMAX*NMAX+4*NMAX+12*MMAX*MMAX+12*MMAX);

    printf ("Hello world\n");
    n=N;
    m=M;
    iprint = 1;
    factr=1.0e+7;
    pgtol=1.0e-5;

    /* Odd numbered fortran variables */
    for (i=0; i < n; i+=2) {
	nbd[i] = 2;
	l[i]=1.0e0;
	u[i]=1.0e2;
    }
    /* Even numbered fortran variables */
    for (i=1; i < n; i+=2) {
	nbd[i] = 2;
	l[i]=-1.0e2;
	u[i]=1.0e2;
    }
    for (i=0; i < n; i++) {
	x[i] = 3.0e0;
    }

    /* Remember: Fortran expects strings to be padded with blanks */
    memset (task, ' ', sizeof(task));
    memcpy (task, "START", 5);
    printf ("%c%c%c%c%c%c%c%c%c%c\n", 
	    task[0], task[1], task[2], task[3], task[4], 
	    task[5], task[6], task[7], task[8], task[9]);

    while (1) {
	setulb_(&n,&m,x,l,u,nbd,&f,g,&factr,&pgtol,wa,iwa,task,&iprint,
		csave,lsave,isave,dsave,60,60);
	for (i = 0; i < 60; i++) printf ("%c", task[i]); printf ("\n");
	if (task[0] == 'F' && task[1] == 'G') {
	    /* Compute cost */
	    t1 = x[0]-1.e0;
	    f = 0.25e0 * t1 * t1;
	    for (i = 1; i < n; i++) {
		t1 = x[i-1] * x[i-1];
		t2 = x[i] - t1;
		f += t2 * t2;
	    }
	    f *= 4.0;
	    printf ("C/f = %g\n", f);
	    /* Compute gradient */
	    t1 = x[1] - x[0] * x[0];
	    g[0] = 2.0 * (x[0] - 1.0) - 1.6e1 * x[0] * t1;
	    for (i = 1; i < n-1; i++) {
		t2 = t1;
		t1 = x[i+1] - x[i] * x[i];
		g[i] = 8.0 * t2 - 1.6e1 * x[i] * t1;
	    }
	    g[n] = 8.0 * t1;
	} else if (s_cmp(task, "NEW_X", (ftnlen)60, (ftnlen)5) == 0) {
	    /* continue */
	} else {
	    break;
	}
    }
}
#endif

void
bspline_optimize_lbfgsb (
		BSPLINE_Xform* bxf, 
		BSPLINE_Parms *parms, 
		Volume *fixed, 
		Volume *moving, 
		Volume *moving_grad,
		FILE* log_fp)
{
    BSPLINE_Score* ssd = &parms->ssd;
    char task[60], csave[60];
    logical lsave[4];
    integer n, m, iprint, *nbd, *iwa, isave[44];
    doublereal f, factr, pgtol, *x, *l, *u, *g, *wa, dsave[29];
    integer i;
    int it = 0;		/* # iterations */
    int fnev = 0;	/* # function evaluations */
    int NMAX, MMAX;
    double best_score;
    int num_to_check;

    NMAX = bxf->num_coeff;
    MMAX = (int) floor (bxf->num_coeff / 100);
    if (MMAX < 20) MMAX = 20;

    logfile_printf (log_fp, "Setting NMAX, MMAX = %d %d\n", NMAX, MMAX);

    nbd = (integer*) malloc (sizeof(integer)*NMAX);
    iwa = (integer*) malloc (sizeof(integer)*3*NMAX);
    x = (doublereal*) malloc (sizeof(doublereal)*NMAX);
    l = (doublereal*) malloc (sizeof(doublereal)*NMAX);
    u = (doublereal*) malloc (sizeof(doublereal)*NMAX);
    g = (doublereal*) malloc (sizeof(doublereal)*NMAX);
    wa = (doublereal*) malloc (sizeof(doublereal)*(2*MMAX*NMAX+4*NMAX+12*MMAX*MMAX+12*MMAX));

    n=NMAX;
    m=MMAX;

    /* If iprint is 1, the file iterate.dat will be created */
    iprint = 1;
    iprint = 0;

    factr=1.0e+7;
    pgtol=1.0e-5;

    /* Bounds for deformation problem */
    for (i=0; i < n; i++) {
	nbd[i] = 0;
	l[i]=-1.0e1;
	u[i]=+1.0e1;
    }

    /* Initial guess */
    for (i=0; i < n; i++) {
	x[i] = bxf->coeff[i];
    }

    /* Remember: Fortran expects strings to be padded with blanks */
    memset (task, ' ', sizeof(task));
    memcpy (task, "START", 5);
    logfile_printf (log_fp, ">>> %c%c%c%c%c%c%c%c%c%c\n", 
	    task[0], task[1], task[2], task[3], task[4], 
	    task[5], task[6], task[7], task[8], task[9]);

    /* run_toy_kernel();
    getchar();
    */

    /* Fill the GPU data structure  */
#if (HAVE_BROOK) && (BUILD_BSPLINE_BROOK)
    if (parms->implementation == BIMPL_BROOK) {
	logfile_printf (log_fp, "Initializing GPU data structures for Brook. \n");
	    // bspline_initialize_structure_to_store_data_from_gpu(fixed, parms);
	bspline_initialize_streams_on_gpu(fixed, moving, moving_grad, bxf, parms);
	logfile_printf (log_fp, "Done. \n");
    }
#endif

#if (HAVE_CUDA) && (BUILD_BSPLINE_CUDA)
	if(parms->implementation == BIMPL_CUDA) {
		bspline_cuda_initialize_d(fixed, moving, moving_grad, bxf, parms);
		// bspline_cuda_initialize(fixed, moving, moving_grad, bxf, parms);
	}
#endif

    while (1) {
	setulb_(&n,&m,x,l,u,nbd,&f,g,&factr,&pgtol,wa,iwa,task,&iprint,
		csave,lsave,isave,dsave,60,60);
	logfile_printf (log_fp, ">>> ");
	for (i = 0; i < 60; i++) {
	    logfile_printf (log_fp, "%c", task[i]);
	}
	logfile_printf (log_fp, "\n");
	if (task[0] == 'F' && task[1] == 'G') {

	    /* Copy from fortran variables (double -> float) */
	    for (i = 0; i < NMAX; i++) {
		bxf->coeff[i] = (float) x[i];
	    }

	    /* Compute cost and gradient */
	    bspline_score (parms, bxf, fixed, moving, moving_grad, log_fp);
	    /* Give a little feedback to the user */
	    bspline_display_coeff_stats (log_fp, bxf);

	    /* Copy to fortran variables (float -> double) */
	    f = ssd->score;
	    for (i = 0; i < NMAX; i++) {
		g[i] = - ssd->grad[i];
	    }

	    /* Check # iterations */
	    if (++fnev == parms->max_its) break;

	} else if (s_cmp(task, "NEW_X", (ftnlen)60, (ftnlen)5) == 0) {
	    /* Optimizer has completed an iteration */
	    /* Check convergence tolerance */
	    if (it == 0) {
		best_score = ssd->score;
		num_to_check = parms->convergence_tol_its;
	    } else {
		if (ssd->score < best_score - parms->convergence_tol) {
		    best_score = ssd->score;
		    num_to_check = parms->convergence_tol_its;
		} else {
		    num_to_check --;
		}
	    }
	    logfile_printf (log_fp, "Score: %g, Best: %g, It: %d\n", ssd->score, best_score, num_to_check);
	    if (num_to_check <= 0) {
		break;
	    }
	    /* else continue */
	} else {
	    break;
	}
    }

#if (HAVE_CUDA) && (BUILD_BSPLINE_CUDA)
	if(parms->implementation == BIMPL_CUDA) {
		bspline_cuda_clean_up_d();
		// bspline_cuda_clean_up();
	}
#endif

    free (nbd);
    free (iwa);
    free (x);
    free (l);
    free (u);
    free (g);
    free (wa);
}
