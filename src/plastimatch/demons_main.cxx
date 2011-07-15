/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plm_config.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "delayload.h"
#include "demons.h"
#include "demons_opts.h"
#include "mha_io.h"
#include "vf.h"
#include "vf_stats.h"
#include "volume.h"

int
main (int argc, char* argv[])
{
    DEMONS_Options options;
    Volume* fixed;
    Volume* moving;
    Volume* warped;
    Volume* moving_grad = 0;
    Volume* vector_field;

    /* Read in command line options */
    parse_args (&options, argc, argv);

    printf("\nReading the static/reference image\n");
    fixed = read_mha (options.fixed_fn);
    if (!fixed) return -1;

    printf("\nReading the moving image \n");
    moving = read_mha (options.moving_fn);
    if (!moving) return -1;

    if (fixed->npix != moving->npix) {
	printf("\nVolumes have different dimensions.....Exiting\n");
	exit(-1);
    }

    /* GCS FIX: This limitation only applies to brook version */
    if (options.parms.threading == THREADING_BROOK) {
	if (fixed->dim[0] % 4 != 0) {
	    printf("\nX dimension must be divisible by 4.  Exiting.\n");
	    exit(-1);
	}
    }

    volume_convert_to_float (moving);
    volume_convert_to_float (fixed);
    if (options.parms.threading != THREADING_OPENCL) {
	moving_grad = volume_make_gradient (moving);
	//write_mha ("moving_grad.mha", moving_grad);
    }

    vector_field = demons (fixed, moving, moving_grad, 0, 
	&options.parms);

    vf_analyze (vector_field);

    if (options.output_vf_fn) {
	write_mha (options.output_vf_fn, vector_field);
    }

    warped = vf_warp (0, moving, vector_field);

    if (options.output_img_fn) {
	write_mha (options.output_img_fn, warped);
    }

    volume_destroy (fixed);
    volume_destroy (moving);
    if (options.parms.threading != THREADING_OPENCL) {
	volume_destroy (moving_grad);
    }
    volume_destroy (vector_field);

    return 0;
}
