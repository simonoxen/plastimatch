/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
/* -------------------------------------------------------------------------
    REFS:
    http://en.wikipedia.org/wiki/B-spline
    http://www.cs.mtu.edu/~shene/COURSES/cs3621/NOTES/surface/bspline-construct.html
    http://graphics.idav.ucdavis.edu/education/CAGDNotes/Quadratic-B-Spline-Surface-Refinement/Quadratic-B-Spline-Surface-Refinement.html

    ----------------------------------------------------------------------- */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "plm_config.h"
#include "volume.h"
#include "readmha.h"
#include "bspline_opts.h"
#include "bspline.h"
#if defined (HAVE_F2C_LIBRARY)
#include "bspline_optimize_lbfgsb.h"
#endif

int
main (int argc, char* argv[])
{
    BSPLINE_Options options;
    BSPLINE_Parms* parms = &options.parms;
    Volume *moving, *fixed, *moving_grad;
    Volume *vector_field;
    Volume *moving_warped;

    parse_args (&options, argc, argv);

    fixed = read_mha (options.fixed_fn);
    moving = read_mha (options.moving_fn);

    volume_convert_to_float (moving);
    volume_convert_to_float (fixed);

    moving_grad = volume_make_gradient (moving);

    /* Debug */
    //write_mha ("moving_grad.mha", moving_grad);

    /* Set parms based on fixed image */
    parms->img_origin[0] = fixed->offset[0];
    parms->img_origin[1] = fixed->offset[1];
    parms->img_origin[2] = fixed->offset[2];
    parms->img_spacing[0] = fixed->pix_spacing[0];
    parms->img_spacing[1] = fixed->pix_spacing[1];
    parms->img_spacing[2] = fixed->pix_spacing[2];
    parms->img_dim[0] = fixed->dim[0];
    parms->img_dim[1] = fixed->dim[1];
    parms->img_dim[2] = fixed->dim[2];
    parms->roi_offset[0] = 0;
    parms->roi_offset[1] = 0;
    parms->roi_offset[2] = 0;
    parms->roi_dim[0] = fixed->dim[0];
    parms->roi_dim[1] = fixed->dim[1];
    parms->roi_dim[2] = fixed->dim[2];

    /* Allocate memory and build lookup tables */
    bspline_initialize (parms);

    /* Run the optimization */
    bspline_optimize (parms, fixed, moving, moving_grad);

    /* Create vector field from bspline coefficients and save */
    vector_field = volume_create (fixed->dim, fixed->offset, fixed->pix_spacing,
			    PT_VF_FLOAT_INTERLEAVED, 0);
    bspline_interpolate_vf (vector_field, parms);
    write_mha (options.output_fn, vector_field);

    /* Create warped output image and save */
    moving_warped = volume_warp (0, moving, vector_field);
    write_mha ("warped.mha", moving_warped);

    /* Free memory */
    bspline_free (parms);
    volume_free (fixed);
    volume_free (moving);
    volume_free (moving_grad);
    volume_free (moving_warped);
    volume_free (vector_field);

    return 0;
}
