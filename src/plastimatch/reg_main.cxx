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


int
main (int argc, char* argv[])
{
    Reg_options options;
    Reg_parms *parms = &options.parms;
    Volume *vf = 0;
    Bspline_xform *bxf = 0;
    float S = 9999.9f;

    reg_opts_parse_args (&options, argc, argv);

    /* algorithm selection */
    if (parms->analytic == true) {
        printf ("Using ANALYTIC method.\n");

        /* Load coeff OR load vf and compute coeff */
        if (options.input_xf_fn == 0) {
            vf = read_mha (options.input_vf_fn);
            if (!vf) { exit (-1); }
            printf ("Computing coefficients from: %s\n", options.input_vf_fn);
            bxf = bxf_from_vf (vf, options.vox_per_rgn);
        } else {
            bxf = bspline_xform_load (options.input_xf_fn);
            if (!bxf) { exit (-1); }
        }

        switch (parms->implementation) {
#if 0
        case 'a':
            /* CALLING FUNCTION FOR ANALYTIC METHOD 'a" GOES HERE */
            S = vf_regularize_analytic (bxf);
            break;
#endif
        default:
            printf ("Sorry, analytic regularization not yet implemented.\n\n");
            exit(0);
            break;
        } /* switch(implementation) */
        bspline_xform_free (bxf);

    } else {
        printf ("Using NUMERICAL method.\n");

        /* Load vf OR load coeff and compute vf */
        if (options.input_vf_fn == 0) {
            bxf = bspline_xform_load (options.input_xf_fn);
            if (!bxf) { exit (-1); }
            printf ("Computing vector field from: %s\n", options.input_xf_fn);
            vf = volume_create (bxf->img_dim, bxf->img_origin, 
                                bxf->img_spacing, 0, 
                                PT_VF_FLOAT_INTERLEAVED, 3, 0);
            bspline_interpolate_vf (vf, bxf);
            bspline_xform_free (bxf);
        } else {
            vf = read_mha (options.input_vf_fn);
            if (!vf) { exit (-1); }
        }

        switch (parms->implementation) {
        case 'a':
            S = vf_regularize_numerical (vf);
            break;
        default:
            printf ("Warning: Using implementation 'a'\n");
            S = vf_regularize_numerical (vf);
            break;
        } /* switch(implementation) */
        volume_destroy (vf);
    }

    printf ("S = %f\n", S);
}
