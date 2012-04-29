/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include <stdio.h>
#include <g2c.h>

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


#define N 25
#define M 5
#define NMAX 1024
#define MMAX 17

void
bspline_optimize (void)
{
    printf ("Hello world\n");

    char task[60], csave[60];
    logical lsave[4];
    integer n, m, iprint, nbd[NMAX], iwa[3*NMAX], isave[44];
    doublereal f, factr, pgtol, x[NMAX], l[NMAX], u[NMAX], g[NMAX],
	    wa[2*MMAX*NMAX+4*NMAX+12*MMAX*MMAX+12*MMAX], dsave[29];

    n=N;
    m=M;

    doublereal t1, t2;
    integer i;

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
