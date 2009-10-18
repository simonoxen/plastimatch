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
    BSPLINE_Xform bxf;
    Volume *moving, *fixed, *moving_grad;
    Volume *vector_field = 0;
    Volume *moving_warped = 0;
    int roi_offset[3];

    bspline_opts_parse_args (&options, argc, argv);

    fixed = read_mha (options. fixed_fn);
    if (!fixed) exit (-1);
    moving = read_mha (options. moving_fn);
    if (!moving) exit (-1);

    volume_convert_to_float (moving);
    volume_convert_to_float (fixed);

    printf ("Making gradient\n");
    moving_grad = volume_make_gradient (moving);

    /* Debug */
    //write_mha ("moving_grad.mha", moving_grad);

    /* Allocate memory and build lookup tables */
    printf ("Allocating lookup tables\n");
    memset (roi_offset, 0, 3*sizeof(int));
    bspline_xform_initialize (&bxf,
			      fixed->offset,
			      fixed->pix_spacing,
			      fixed->dim,
			      roi_offset,
			      fixed->dim,
			      options.vox_per_rgn
			     );

    /* Run the optimization */
    printf ("Running optimization.\n");
    bspline_optimize (&bxf, 0, parms, fixed, moving, moving_grad);
    printf ("Done running optimization.\n");

    /* Save output transform */
    if (options.output_xf_fn) {
	write_bxf (options.output_xf_fn, &bxf);
    }

    /* Create vector field from bspline coefficients and save */
    if (options.output_vf_fn || options.output_warped_fn) {
	printf ("Creating vector field.\n");
	vector_field = volume_create (fixed->dim, fixed->offset, 
				      fixed->pix_spacing,
				      PT_VF_FLOAT_INTERLEAVED, 
				      fixed->direction_cosines, 0);
	bspline_interpolate_vf (vector_field, &bxf);
	if (options.output_vf_fn) {
	    printf ("Writing vector field.\n");
	    write_mha (options.output_vf_fn, vector_field);
	}
    }
	
    //printf("%f",moving_warped->dim[1]);
    /* Create warped output image and save */
    if (options.output_warped_fn) {
	printf ("Warping image.\n");
	moving_warped = volume_warp (0, moving, vector_field);
	printf ("Writing warped image.\n");
	//printf("write to %s\n",options.output_warped_fn);
	//system("pause");
	write_mha (options.output_warped_fn, moving_warped);
    }

    /* Free memory */
    printf ("Done warping images.\n");
    bspline_parms_free (parms);
    bspline_xform_free (&bxf);
    volume_free (fixed);
    volume_free (moving);
    volume_free (moving_grad);
    volume_free (moving_warped);
    volume_free (vector_field);

    printf ("Done freeing memory\n");

    return 0;
}
