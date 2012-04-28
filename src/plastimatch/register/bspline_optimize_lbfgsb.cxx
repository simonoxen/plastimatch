/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plm_config.h"
#include <limits>
#include <float.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "bspline.h"
#include "bspline_optimize.h"
#include "bspline_optimize_lbfgsb.h"
#include "bspline_opts.h"
#include "plm_fortran.h"

#include "plmbase.h"
#include "plmsys.h"

/* EXTERNAL DEPENDS */
#include "bspline_xform.h"
#include "plm_math.h"
#include "volume.h"


/* Defined in f2c.h, conflicts with std::numeric_limits::max */
#if defined max
#undef max
#endif

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

class Nocedal_optimizer
{
public:
    char task[60], csave[60];
    logical lsave[4];
    integer n, m, iprint, *nbd, *iwa, isave[44];
    doublereal f, factr, pgtol, *x, *l, *u, *g, *wa, dsave[29];
public:
    Nocedal_optimizer (Bspline_optimize_data *bod);
    ~Nocedal_optimizer () {
        free (nbd);
        free (iwa);
        free (x);
        free (l);
        free (u);
        free (g);
        free (wa);
    }
    void setulb () {
        setulb_ (&n,&m,x,l,u,nbd,&f,g,&factr,&pgtol,wa,iwa,task,&iprint,
            csave,lsave,isave,dsave,60,60);
    }
};

Nocedal_optimizer::Nocedal_optimizer (Bspline_optimize_data *bod)
{
    Bspline_xform *bxf = bod->bxf;
    Bspline_parms *parms = bod->parms;

    int nmax = bxf->num_coeff;
    int mmax = 20;

    /* Try to allocate memory for hessian approximation.  
       First guess based on heuristic. */
    if (bxf->num_coeff >= 20) {
        mmax = 20 + (int) floor (sqrt ((float) (bxf->num_coeff - 20)));
        if (mmax > std::numeric_limits<int>::max() / nmax / 10) {
            mmax = std::numeric_limits<int>::max() / nmax / 10;
        }
        if (mmax > 500) {
            mmax = 500;
        }
    }
    do {
        nbd = (integer*) malloc (sizeof(integer)*nmax);
        iwa = (integer*) malloc (sizeof(integer)*3*nmax);
        x = (doublereal*) malloc (sizeof(doublereal)*nmax);
        l = (doublereal*) malloc (sizeof(doublereal)*nmax);
        u = (doublereal*) malloc (sizeof(doublereal)*nmax);
        g = (doublereal*) malloc (sizeof(doublereal)*nmax);
        wa = (doublereal*) malloc (sizeof(doublereal)
            *(2*mmax*nmax+4*nmax+12*mmax*mmax+12*mmax));

        if ((nbd == NULL) ||
            (iwa == NULL) ||
            (  x == NULL) ||
            (  l == NULL) ||
            (  u == NULL) ||
            (  g == NULL) ||
            ( wa == NULL))
        {
            /* We didn't get enough memory.  Free what we got. */
            free (nbd);
            free (iwa);
            free (x);
            free (l);
            free (u);
            free (g);
            free (wa);

            /* Give a little feedback to the user */
            logfile_printf (
                "Tried nmax, mmax = %d %d, but ran out of memory!\n",
                nmax, mmax);

            /* Try again with reduced request */
            if (mmax > 20) {
                mmax = mmax / 2;
            } else if (mmax > 10) {
                mmax = 10;
            } else if (mmax > 2) {
                mmax --;
            } else {
                print_and_exit ("System ran out of memory when "
                    "initializing Nocedal optimizer.\n");
            }
        }
        else {
            /* Everything went great.  We got the memory. */
            break;
        }
    } while (1);
    m = mmax;
    n = nmax;

    /* Give a little feedback to the user */
    logfile_printf ("Setting nmax, mmax = %d %d\n", nmax, mmax);

    /* If iprint is 1, the file iterate.dat will be created */
    iprint = 0;

    //factr = 1.0e+7;
    //pgtol = 1.0e-5;
    factr = parms->lbfgsb_factr;
    pgtol = parms->lbfgsb_pgtol;

    /* Bounds for deformation problem */
    for (int i = 0; i < n; i++) {
        nbd[i] = 0;
        l[i]=-1.0e1;
        u[i]=+1.0e1;
    }

    /* Initial guess */
    for (int i = 0; i < n; i++) {
        x[i] = bxf->coeff[i];
    }

    /* Remember: Fortran expects strings to be padded with blanks */
    memset (task, ' ', sizeof(task));
    memcpy (task, "START", 5);
    logfile_printf (">>> %c%c%c%c%c%c%c%c%c%c\n", 
        task[0], task[1], task[2], task[3], task[4], 
        task[5], task[6], task[7], task[8], task[9]);
}

void
bspline_optimize_lbfgsb (
    Bspline_optimize_data *bod
)
{
    Bspline_xform *bxf = bod->bxf;
    Bspline_state *bst = bod->bst;
    Bspline_parms *parms = bod->parms;
    Bspline_score* ssd = &bst->ssd;
    FILE *fp = 0;
    double best_score = DBL_MAX;
    float *best_coeff = (float*) malloc (sizeof(float) * bxf->num_coeff);

    Nocedal_optimizer optimizer (bod);

    /* Initialize # iterations, # function evaluations */
    bst->it = 0;
    bst->feval = 0;

    if (parms->debug) {
        fp = fopen ("scores.txt", "w");
    }

    while (1) {
        /* Get next search location */
        optimizer.setulb ();

        if (optimizer.task[0] == 'F' && optimizer.task[1] == 'G') {
            /* Got a new probe location within a line search */

            /* Copy from fortran to C (double -> float) */
            for (int i = 0; i < bxf->num_coeff; i++) {
                bxf->coeff[i] = (float) optimizer.x[i];
            }

            /* Compute cost and gradient */
            bspline_score (bod);

            /* Save coeff if best score */
            if (ssd->score < best_score) {
                best_score = ssd->score;
                for (int i = 0; i < bxf->num_coeff; i++) {
                    best_coeff[i] = bxf->coeff[i];
                }
            }

            /* Give a little feedback to the user */
            bspline_display_coeff_stats (bxf);

            /* Save some debugging information */
            bspline_save_debug_state (parms, bst, bxf);
            if (parms->debug) {
                fprintf (fp, "%f\n", ssd->score);
            }

            /* Copy from C to fortran (float -> double) */
            optimizer.f = ssd->score;
            for (int i = 0; i < bxf->num_coeff; i++) {
                optimizer.g[i] = ssd->grad[i];
            }

            /* Check # feval */
            if (bst->feval >= parms->max_feval) break;
            bst->feval ++;

        } else if (memcmp (optimizer.task, "NEW_X", strlen ("NEW_X")) == 0) {
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

    /* Copy out the best results */
    for (int i = 0; i < bxf->num_coeff; i++) {
        bxf->coeff[i] = best_coeff[i];
    }
    free (best_coeff);
}