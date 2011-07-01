/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plm_config.h"
#include <stdlib.h>
#include <stdio.h>
#include "mha_io.h"
#include "volume.h"
#include "bspline_xform.h"
#include "bspline.h"
#include "reg_opts.h"
#include "reg.h"

#define NUMERICAL 0
#define ANALYTIC  1


Bspline_xform*
bxf_from_vf (Volume* vf, int* vox_per_rgn)
{
    Bspline_xform* bxf;
    int roi_offset[3] = {0, 0, 0};

    bxf = (Bspline_xform*) malloc (sizeof (Bspline_xform));
    bspline_xform_initialize (
        bxf,
        vf->offset,
        vf->spacing,
        vf->dim,
        roi_offset,
        vf->dim,
        vox_per_rgn
    );
    compute_coeff_from_vf (bxf, vf);

    volume_destroy (vf);
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
            vf = compute_vf_from_coeff (bxf);
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

int
main (int argc, char* argv[])
{
    Reg_options options;
    Reg_parms *parms = &options.parms;
    Volume *vf = NULL;
    Bspline_xform *bxf = NULL;
    float S = 9999.9f;

    reg_opts_parse_args (&options, argc, argv);

    /* algorithm selection */
    switch (parms->implementation) {
    case 'a':
        vf = (Volume*)load (&options, NUMERICAL);
        S = vf_regularize_numerical (vf);
        break;
    case 'b':
        bxf = (Bspline_xform*)load (&options, ANALYTIC);
        S = vf_regularize_analytic (bxf);
        break;
    default:
        printf ("Warning: Using implementation 'a'\n");
        vf = (Volume*)load (&options, NUMERICAL);
        S = vf_regularize_numerical (vf);
        break;
    } /* switch(implementation) */

    if (vf) {
        volume_destroy (vf);
    }

    if (bxf) {
        bspline_xform_free (bxf);
    }

    printf ("S = %f\n", S);

    return 0;
}
