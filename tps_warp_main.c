/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plm_config.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "mathutil.h"
#include "readmha.h"
#include "tps_warp_opts.h"

int
main (int argc, char *argv[])
{
    Tps_options options;
    Tps_xform *tps;
    Volume *moving;
    Volume *vf_out = 0;
    Volume *warped_out = 0;

    tps_warp_opts_parse_args (&options, argc, argv);

    printf ("Loading xform\n");
    tps = tps_xform_load (options.tps_xf_fn);
    if (!tps) exit (-1);

    printf ("Reading mha\n");
    moving = read_mha (options.moving_fn);
    if (!moving) exit (-1);

    printf ("Converting volume to float\n");
    volume_convert_to_float (moving);

    if (options.output_vf_fn) {
	printf ("Creating output vf\n");
	vf_out = volume_create (
	    tps->img_dim, 
	    tps->img_origin, 
	    tps->img_spacing, 
	    PT_VF_FLOAT_INTERLEAVED, 
	    0, 0);
    } else {
	vf_out = 0;
    }
    if (options.output_warped_fn) {
	printf ("Creating output vol\n");
	warped_out = volume_create (
	    tps->img_dim, 
	    tps->img_origin, 
	    tps->img_spacing, 
	    PT_FLOAT, 
	    0, 0);
    } else {
	warped_out = 0;
    }
	
    printf ("Calling tps_warp...\n");
    tps_warp (warped_out, vf_out, tps, moving, 1, -1000);
    printf ("done!\n");

    if (options.output_vf_fn) {
	printf ("Writing output vf.\n");
	write_mha (options.output_vf_fn, vf_out);
    }
    if (options.output_warped_fn) {
	printf ("Writing output vol.\n");
	write_mha (options.output_warped_fn, warped_out);
    }

    printf ("Finished.\n");
    return 0;
}
