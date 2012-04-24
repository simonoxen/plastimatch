/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plm_config.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include "libplmimage.h"

#include "bspline.h"
#include "bspline_xform.h"
#include "bspline_regularize_analytic.h"
#include "bspline_regularize_numeric.h"
#include "bspline_regularize_state.h"
#include "plm_timer.h"
#include "reg_opts.h"
#include "volume.h"

#define NUMERICAL 0
#define ANALYTIC  1


void
print_stats (Bspline_xform* bxf, Bspline_score* bscore, float score, double time)
{
    int i;
    float grad_mean = 0.0f;
    float grad_norm = 0.0f;

    printf ("%9.3f   [ %9.3f s ]\n", score, time);

    if (bscore->grad) {
        for (i=0; i<bxf->num_coeff; i++) {
            grad_mean += bscore->grad[i];
            grad_norm += fabs (bscore->grad[i]);
        }
        printf ("GM %9.3f   GN %9.3f\n", grad_mean, grad_norm);
    }
}

Bspline_xform*
bxf_from_vf (Volume* vf, plm_long* vox_per_rgn)
{
    Bspline_xform* bxf;
    plm_long roi_offset[3] = {0, 0, 0};

    bxf = (Bspline_xform*) malloc (sizeof (Bspline_xform));
    bspline_xform_initialize (
        bxf,
        vf->offset,
        vf->spacing,
        vf->dim,
        roi_offset,
        vf->dim,
        vox_per_rgn,
        (vf->direction_cosines).m_direction_cosines
    );
    compute_coeff_from_vf (bxf, vf);

    delete vf;
    return bxf;
}

void*
load (Reg_options* options, int mode)
{
    Bspline_xform* bxf = NULL;
    Volume* vf = NULL;

    /* Numerical Mode */
    if (mode == NUMERICAL) {
        /* vf volume not supplied: load coeff and compute vf */
        if (options->input_vf_fn == 0) {
            bxf = bspline_xform_load (options->input_xf_fn);
            if (!bxf) { exit (-1); }
            printf ("Computing vector field from: %s\n", options->input_xf_fn);
            vf = bspline_compute_vf (bxf);
            bspline_xform_free (bxf);
        /* Simply load vf volume if supplied */
        } else {
            vf = read_mha (options->input_vf_fn);
            if (!vf) { exit (-1); }
        }
        return (void*)vf;
    }
    /* Analytic Mode */
    else if (mode == ANALYTIC) {
        /* bxf not supplied: load vf and compute coeff */
        if (options->input_xf_fn == 0) {
            vf = read_mha (options->input_vf_fn);
            if (!vf) { exit (-1); }
            printf ("Computing coefficients from: %s\n", options->input_vf_fn);
            bxf = bxf_from_vf (vf, options->vox_per_rgn);
        } else {
            bxf = bspline_xform_load (options->input_xf_fn);
            if (!bxf) { exit (-1); }
        }
        return (void*)bxf;
    }
    /* Invalid Mode */
    else {
        fprintf (stderr, "Internal Error: invalid load mode\n");
        return NULL;
    }

}

void
init_bscore (Bspline_xform* bxf, Bspline_score* ssd)
{
    ssd->grad = (float*)malloc (bxf->num_coeff * sizeof (float));
    memset (ssd->grad, 0, bxf->num_coeff * sizeof (float));
    ssd->rmetric = 0;
}

int
main (int argc, char* argv[])
{
    Plm_timer timer;
    double time;
    Reg_options options;
    Reg_parms *parms = &options.parms;
    Bspline_regularize_state rst;
    Volume *vf = NULL;
    Bspline_score bscore;
    Bspline_xform *bxf = NULL;
    float S = 9999.9f;

    reg_opts_parse_args (&options, argc, argv);

    parms->lambda = 1.0f;

    /* algorithm selection */
    switch (parms->implementation) {
    default:
        printf ("Warning: Using implementation 'a'\n");
	/* Fall through */
    case 'a':
#if defined (commentout)
        vf = (Volume*) load (&options, NUMERICAL);
        plm_timer_start (&timer);
        S = vf_regularize_numerical (vf);
        time = plm_timer_report (&timer);
        break;
#endif
        bxf = (Bspline_xform*) load (&options, ANALYTIC);
        init_bscore (bxf, &bscore);

        plm_timer_start (&timer);
        bspline_regularize_numeric_a_init (&rst, bxf);
        bspline_regularize_numeric_a (&bscore, parms, &rst, bxf);
        bspline_regularize_numeric_a_destroy (&rst, bxf);
        time = plm_timer_report (&timer);

        S = bscore.rmetric;
        break;
    case 'b':
        bxf = (Bspline_xform*) load (&options, ANALYTIC);
        init_bscore (bxf, &bscore);

        plm_timer_start (&timer);
        vf_regularize_analytic_init (&rst, bxf);
        vf_regularize_analytic (&bscore, parms, &rst, bxf);
        vf_regularize_analytic_destroy (&rst);
        time = plm_timer_report (&timer);

        S = bscore.rmetric;
        break;
    case 'c':
#if (OPENMP_FOUND)
        bxf = (Bspline_xform*) load (&options, ANALYTIC);
        init_bscore (bxf, &bscore);

        plm_timer_start (&timer);
        vf_regularize_analytic_init (&rst, bxf);
        vf_regularize_analytic_omp (&bscore, parms, &rst, bxf);
        vf_regularize_analytic_destroy (&rst);
        time = plm_timer_report (&timer);

        S = bscore.rmetric;
#else
        printf ("OpenMP is required to use implementation (c).\n");
        printf ("Exiting...\n\n");
        exit (0);
#endif
        break;
    case 'd':
        bxf = (Bspline_xform*) load (&options, ANALYTIC);
        init_bscore (bxf, &bscore);

        plm_timer_start (&timer);
	bspline_regularize_numeric_d_init (&rst, bxf);
        bspline_regularize_numeric_d (&bscore, parms, &rst, bxf);
	bspline_regularize_numeric_d_destroy (&rst, bxf);
        time = plm_timer_report (&timer);

        S = bscore.score;
        break;
    } /* switch(implementation) */

    printf ("Printing stats.\n");
    print_stats (bxf, &bscore, S, time);

    if (vf) {
        delete vf;
    }

    if (bxf) {
        bspline_xform_free (bxf);
        free (bscore.grad);
    }

    return 0;
}
