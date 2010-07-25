/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plm_config.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "math_util.h"
#include "mha_io.h"
#include "landmark_warp_opts.h"
#include "print_and_exit.h"

/* How do the algorithms load their point data (currently)?
   plastimatch warp pointset   - PointSetType
   itk_tps_warp                - TPS_parms + PointSetType (equivalent)
   GCS (landmark warp)         - Tps_xform (includes alpha, alpha * dist)
   NSH (bspline_rbf)           - Bspline_landmarks (includes warped lm, others)
*/
static void
load_landmarks (Landmark_warp_options *options)
{
    if (options->input_xform_fn)
    {
	/* Load as xform */
    }
    else if (options->input_fixed_landmarks_fn 
	&& options->input_moving_landmarks_fn)
    {
	/* Load as raw pointsets */
    }
    else
    {
	print_and_exit (
	    "Sorry, you must specify either an xform file or \n"
	    "fixed and moving landmark files.");
    }
}

static void
do_landmark_warp_gcs (Landmark_warp_options *options)
{
    Tps_xform *tps;
    Volume *moving;
    Volume *vf_out = 0;
    Volume *warped_out = 0;

    printf ("Loading xform\n");
    tps = tps_xform_load (options->input_xform_fn);
    if (!tps) exit (-1);

    printf ("Reading mha\n");
    moving = read_mha (options->input_moving_image_fn);
    if (!moving) exit (-1);

    printf ("Converting volume to float\n");
    volume_convert_to_float (moving);

    if (options->output_vf_fn) {
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
    if (options->output_warped_image_fn) {
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
    if (options->output_vf_fn) {
	printf ("Writing output vf.\n");
	write_mha (options->output_vf_fn, vf_out);
    }
    if (options->output_warped_image_fn) {
	printf ("Writing output vol.\n");
	write_mha (options->output_warped_image_fn, warped_out);
    }

    printf ("Finished.\n");
}

static void
do_landmark_warp (Landmark_warp_options *options)
{
    load_landmarks (options);

    switch (options->algorithm) {
    case LANDMARK_WARP_ALGORITHM_ITK_TPS:
	break;
    case LANDMARK_WARP_ALGORITHM_RBF_NSH:
	break;
    case LANDMARK_WARP_ALGORITHM_RBF_GCS:
    default:
	do_landmark_warp_gcs (options);
	break;
    }
}

int
main (int argc, char *argv[])
{
    Landmark_warp_options options;

    landmark_warp_opts_parse_args (&options, argc, argv);
    do_landmark_warp (&options);
    return 0;
}
