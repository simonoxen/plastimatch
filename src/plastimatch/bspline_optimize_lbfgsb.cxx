/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plm_config.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include "bspline.h"
#include "bspline_optimize.h"
#include "bspline_optimize_lbfgsb.h"
#include "bspline_opts.h"
#include "logfile.h"
#include "plm_fortran.h"
#include "volume.h"
#include "print_and_exit.h"

#if defined __cplusplus
extern "C" {
#endif
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
#if defined __cplusplus
}
#endif

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
	    //	} else if (s_cmp(task, "NEW_X", (ftnlen)60, (ftnlen)5) == 0) {
	} else if (memcmp (task, "NEW_X", strlen ("NEW_X")) == 0) {
	    /* continue */
	} else {
	    break;
	}
    }
}
#endif

void
bspline_optimize_lbfgsb (
    Bspline_optimize_data *bod
)
{
    Bspline_xform *bxf = bod->bxf;
    Bspline_state *bst = bod->bst;
    Bspline_parms *parms = bod->parms;
    Volume *fixed = bod->fixed;
    Volume *moving = bod->moving;
    Volume *moving_grad = bod->moving_grad;
    Bspline_score* ssd = &bst->ssd;
    char task[60], csave[60];
    logical lsave[4];
    integer n, m, iprint, *nbd, *iwa, isave[44];
    doublereal f, factr, pgtol, *x, *l, *u, *g, *wa, dsave[29];
    integer i;
    int NMAX, MMAX;
    FILE *fp = 0;
    //    double best_score;

    /* F2C Builtin function */
    //    integer s_cmp (char *, char *, ftnlen, ftnlen);

    NMAX = bxf->num_coeff;
    // MMAX = (int) floor (bxf->num_coeff / 100);

    /* GCS: Sep 29, 2009.  The previous rule overflows.  I hacked the 
       following new rule, which does not overflow, but is not 
       tested, and may be worse for practical cases. */
    if (bxf->num_coeff < 20) {
	MMAX = 20;
    } else {
	MMAX = 20 + (int) floor (sqrt ((float) (bxf->num_coeff - 20)));
	// MMAX = (int) floor (bxf->num_coeff / 100);
	if (MMAX > 1000) {
	    MMAX = 1000;
	}
    }

    logfile_printf ("Setting NMAX, MMAX = %d %d\n", NMAX, MMAX);

    nbd = (integer*) malloc (sizeof(integer)*NMAX);
    iwa = (integer*) malloc (sizeof(integer)*3*NMAX);
    x = (doublereal*) malloc (sizeof(doublereal)*NMAX);
    l = (doublereal*) malloc (sizeof(doublereal)*NMAX);
    u = (doublereal*) malloc (sizeof(doublereal)*NMAX);
    g = (doublereal*) malloc (sizeof(doublereal)*NMAX);
    wa = (doublereal*) malloc (sizeof(doublereal)*(2*MMAX*NMAX+4*NMAX+12*MMAX*MMAX+12*MMAX));

    if ( (nbd == NULL) ||
         (iwa == NULL) ||
         (  x == NULL) ||
         (  l == NULL) ||
         (  u == NULL) ||
         (  g == NULL) ||
         ( wa == NULL) )
    {
        error_printf ("System ran out of memory when initializing optimizer.\n\n");
        exit (1);
    }

    n=NMAX;
    m=MMAX;

    /* If iprint is 1, the file iterate.dat will be created */
    iprint = 1;
    iprint = 0;

    //factr = 1.0e+7;
    //pgtol = 1.0e-5;
    factr = parms->lbfgsb_factr;
    pgtol = parms->lbfgsb_pgtol;

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
    logfile_printf (">>> %c%c%c%c%c%c%c%c%c%c\n", 
		    task[0], task[1], task[2], task[3], task[4], 
		    task[5], task[6], task[7], task[8], task[9]);

    /* Initialize # iterations, # function evaluations */
    bst->it = 0;
    bst->feval = 0;

    if (parms->debug) {
	fp = fopen ("scores.txt", "w");
    }

    while (1) {
	setulb_(&n,&m,x,l,u,nbd,&f,g,&factr,&pgtol,wa,iwa,task,&iprint,
		csave,lsave,isave,dsave,60,60);
#if defined (commentout)
	logfile_printf (">>> ");
	for (i = 0; i < 60; i++) {
	    logfile_printf ("%c", task[i]);
	}
	logfile_printf ("\n");
#endif
	if (task[0] == 'F' && task[1] == 'G') {

	    /* Copy from fortran to C (double -> float) */
	    for (i = 0; i < NMAX; i++) {
		bxf->coeff[i] = (float) x[i];
	    }

	    /* Compute cost and gradient */
	    bspline_score (parms, bst, bxf, fixed, moving, moving_grad);

	    /* Give a little feedback to the user */
	    bspline_display_coeff_stats (bxf);
	    /* Save some debugging information */
	    bspline_save_debug_state (parms, bst, bxf);
	    if (parms->debug) {
		fprintf (fp, "%f\n", ssd->score);
	    }

	    /* Copy from C to fortran (float -> double) */
	    f = ssd->score;
	    for (i = 0; i < NMAX; i++) {
		g[i] = ssd->grad[i];
	    }

	    /* Check # feval */
	    if (bst->feval >= parms->max_feval) break;
	    bst->feval ++;

	} else if (memcmp (task, "NEW_X", strlen ("NEW_X")) == 0) {
	    /* Optimizer has completed a line search. */

	    /* Check iterations */
	    if (bst->it >= parms->max_its) break;
	    bst->it ++;

	} else {
	    break;
	}
    }

    if (parms->debug) {
	fclose (fp);
    }

    free (nbd);
    free (iwa);
    free (x);
    free (l);
    free (u);
    free (g);
    free (wa);
}
