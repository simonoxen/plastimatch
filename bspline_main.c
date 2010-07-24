/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
/* -------------------------------------------------------------------------
    REFS:
    http://en.wikipedia.org/wiki/B-spline
    http://www.cs.mtu.edu/~shene/COURSES/cs3621/NOTES/surface/bspline-construct.html
    http://graphics.idav.ucdavis.edu/education/CAGDNotes/Quadratic-B-Spline-Surface-Refinement/Quadratic-B-Spline-Surface-Refinement.html
    ----------------------------------------------------------------------- */
#include "plm_config.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "bspline.h"
#include "bspline_landmarks.h"
#include "bspline_rbf.h"
#if defined (HAVE_F2C_LIBRARY)
#include "bspline_optimize_lbfgsb.h"
#endif
#include "bspline_opts.h"
#if (CUDA_FOUND)
#include "bspline_cuda.h"
#endif
#include "mha_io.h"
#include "vf.h"

int
main (int argc, char* argv[])
{
    BSPLINE_Options options;
    Bspline_parms *parms = &options.parms;
    Bspline_xform *bxf;
    Volume *moving, *fixed, *moving_grad;
    Volume *vector_field = 0;
    Volume *moving_warped = 0;
    int roi_offset[3];

    bspline_opts_parse_args (&options, argc, argv);

#if (CUDA_FOUND)
    CUDA_selectgpu (parms->gpuid);
#endif

    fixed = read_mha (options.fixed_fn);
    if (!fixed) exit (-1);
    moving = read_mha (options.moving_fn);
    if (!moving) exit (-1);

    volume_convert_to_float (moving);
    volume_convert_to_float (fixed);

    printf ("Making gradient\n");
    moving_grad = volume_make_gradient (moving);

    /* Load and adjust landmarks */
    if (options.fixed_landmarks && options.moving_landmarks) {
	parms->landmarks = bspline_landmarks_load (
	    options.fixed_landmarks, options.moving_landmarks);
	bspline_landmarks_adjust (parms->landmarks, fixed, moving);
    }

    /* Debug */
    //write_mha ("moving_grad.mha", moving_grad);

    /* Allocate memory and build lookup tables */
    printf ("Allocating lookup tables\n");
    memset (roi_offset, 0, 3*sizeof(int));
    if (options.input_xf_fn) {
	bxf = read_bxf (options.input_xf_fn);
    } else {
	bxf = (Bspline_xform*) malloc (sizeof (Bspline_xform));
	bspline_xform_initialize (
	    bxf,
	    fixed->offset,
	    fixed->pix_spacing,
	    fixed->dim,
	    roi_offset,
	    fixed->dim,
	    options.vox_per_rgn
	);
    }

    /* Run the optimization */
    printf ("Running optimization.\n");
    bspline_run_optimization (bxf, 0, parms, fixed, moving, moving_grad);
    printf ("Done running optimization.\n");

    /* Save output transform */
    if (options.output_xf_fn) {
	write_bxf (options.output_xf_fn, bxf);
    }

    /* Create vector field from bspline coefficients and save */
    if (options.output_vf_fn 
	|| options.output_warped_fn 
	|| (options.warped_landmarks && options.fixed_landmarks 
	    && options.moving_landmarks))
    {
	printf ("Creating vector field.\n");
	vector_field = volume_create (fixed->dim, fixed->offset, 
	    fixed->pix_spacing,
	    PT_VF_FLOAT_INTERLEAVED, 
	    fixed->direction_cosines, 0);
	bspline_interpolate_vf (vector_field, bxf);
    }

    /* Assuming vector field has been created, update warped landmarks*/
    if (options.warped_landmarks && options.fixed_landmarks 
	&& options.moving_landmarks)
    {
	printf ("Creating warped landmarks\n");
	bspline_landmarks_warp (vector_field, parms, bxf, fixed, moving);
	if (options.warped_landmarks) {
	    printf("Writing warped landmarks\n");
	    bspline_landmarks_write_file( options.warped_landmarks, "warped", 
		parms->landmarks->warped_landmarks, 
		parms->landmarks->num_landmarks );
	}
    }

    /* If using radial basis functions, find coeffs and update vector field */
    if (parms->rbf_radius>0) {
	printf ("Radial basis functions requested, radius %.2f\n", 
	    parms->rbf_radius);
	if (!vector_field) {
	    printf ("Sorry, vector field must be present for RBF. Please use -O or -V\n");
	} else {
	    printf ("Warping image before RBF.\n");
	    moving_warped = vf_warp (0, moving, vector_field);
	    write_mha ("wnorbf.mha", moving_warped);
	    if ( parms->landmarks->num_landmarks < 1 ) {
		printf("Sorry, no landmarks found\n");
	    } else {
		/* Do actual RBF adjustment */
		bspline_rbf_find_coeffs( vector_field, parms );
		bspline_rbf_update_vector_field( vector_field, parms );
		bspline_landmarks_warp (vector_field, parms, bxf, 
		    fixed, moving);
		if (options.warped_landmarks)
		    bspline_landmarks_write_file (
			options.warped_landmarks, "warp_and_rbf", 
			parms->landmarks->warped_landmarks, 
			parms->landmarks->num_landmarks);
	    }
	}
    }

    /* Create warped output image and save */
    if (options.output_warped_fn) {
	printf ("Warping image.\n");
	moving_warped = vf_warp (0, moving, vector_field);
	if (moving_warped) {
	    printf ("Writing warped image.\n");
	    write_mha (options.output_warped_fn, moving_warped);
	} else {
	    printf ("Sorry, couldn't create warped image.\n");
	}
    }

    /* Write the vector field */
    if (options.output_vf_fn) {
	printf ("Writing vector field.\n");
	write_mha (options.output_vf_fn, vector_field);
    }

    /* Free memory */
    printf ("Done warping images.\n");
    bspline_parms_free (parms);
    bspline_xform_free (bxf);
    free (bxf);
    volume_destroy (fixed);
    volume_destroy (moving);
    volume_destroy (moving_grad);
    volume_destroy (moving_warped);
    volume_destroy (vector_field);

    printf ("Done freeing memory\n");

    return 0;
}
