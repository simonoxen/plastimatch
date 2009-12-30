/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "plm_config.h"
#include "demons.h"
#include "demons_opts.h"
#include "readmha.h"
#include "vf.h"
#include "volume.h"

int
main (int argc, char* argv[])
{
    DEMONS_Options options;
    Volume* fixed;
    Volume* moving;
    Volume* warped;
    Volume* moving_grad;
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
    if (fixed->dim[0] % 4 != 0) {
	printf("\nX dimension must be divisible by 4.  Exiting.\n");
	exit(-1);
    }

    volume_convert_to_float (moving);
    volume_convert_to_float (fixed);
    moving_grad = volume_make_gradient (moving);
    //write_mha ("moving_grad.mha", moving_grad);

    vector_field = demons (fixed, moving, moving_grad, 0, options.method, &options.parms);

    vf_print_stats (vector_field);

    write_mha ("demons_vf.mha", vector_field);

    warped = vf_warp (0, moving, vector_field);

    write_mha ("warped.mha", warped);

    volume_destroy (fixed);
    volume_destroy (moving);
    volume_destroy (moving_grad);
    volume_destroy (vector_field);

    return 0;
}
