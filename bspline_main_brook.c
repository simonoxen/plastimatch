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
#include "config.h"
#include "volume.h"
#include "readmha.h"
#include "bspline_opts.h"
#include "bspline.h"
#if defined (HAVE_F2C_LIBRARY)
#include "bspline_optimize_lbfgsb.h"
#endif

void testme ();


/* This version of the code only deformations the 3-D volume in X */
int
main (int argc, char* argv[])
{
    BSPLINE_Options options;
    BSPLINE_Data bspd;
    BSPLINE_Score ssd;
    Volume *moving, *fixed, *moving_grad;
    Volume *interp;
    
    parse_args (&options, argc, argv);

    fixed = read_mha (options.fixed_fn);
    moving = read_mha (options.moving_fn);

    volume_convert_to_float (moving);
    volume_convert_to_float (fixed);

    moving_grad = volume_make_gradient (moving);

    bspline_initialize (&bspd, fixed, &options);
    ssd.grad = (float*) malloc (bspd.num_knots * sizeof(float));

    if (options.algorithm == BA_LBFGSB) {
#if defined (HAVE_F2C_LIBRARY)
	bspline_optimize_lbfgsb (&ssd, fixed, moving, moving_grad, 
				 &bspd, &options);
#else
	print_usage ();
#endif
    } else {
	bspline_optimize_steepest (&ssd, fixed, moving, moving_grad, 
				   &bspd, &options);
    }

    /* Create output file of vector field */
    interp = volume_create (fixed->dim, fixed->offset, fixed->pix_spacing,
			    PT_VF_FLOAT_INTERLEAVED);
    bspline_interpolate (interp, &bspd, &options);
    write_mha (options.output_fn, interp);

    return 0;
}
