/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plm_config.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#include "bstring_util.h"
#include "math_util.h"
#include "mha_io.h"
#include "landmark_warp.h"
#include "landmark_warp_args.h"
#include "landmark_warp_ggo.h"
#include "plm_ggo.h"
#include "pointset.h"
#include "print_and_exit.h"
#include "rbf_gcs.h"

/* How do the algorithms load their point data (currently)?
   plastimatch warp pointset   - PointSetType
                                 itk_pointset.h
   itk_tps_warp                - TPS_parms + PointSetType (equivalent)
                                 itk_tps.h
   GCS (landmark warp)         - Tps_xform (includes alpha, alpha * dist)
                                 tps.h
   NSH (bspline_rbf)           - Bspline_landmarks (includes warped lm, others)
                                 bspline_landmarks.h
*/

static void
do_landmark_warp_itk_tps (Landmark_warp *lw)
{
    
}

#if defined (commentout)
static void
do_landmark_warp_gcs (Landmark_warp_args *parms)
{
    Tps_xform *tps;
    Volume *moving;
    Volume *vf_out = 0;
    Volume *warped_out = 0;

    printf ("Loading xform\n");
    tps = tps_xform_load (parms->input_xform_fn);
    if (!tps) exit (-1);

    printf ("Reading mha\n");
    moving = read_mha (parms->input_moving_image_fn);
    if (!moving) exit (-1);

    printf ("Converting volume to float\n");
    volume_convert_to_float (moving);

    if (parms->output_vf_fn) {
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
    if (parms->output_warped_image_fn) {
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
    if (parms->output_vf_fn) {
	printf ("Writing output vf.\n");
	write_mha (parms->output_vf_fn, vf_out);
    }
    if (parms->output_warped_image_fn) {
	printf ("Writing output vol.\n");
	write_mha (parms->output_warped_image_fn, warped_out);
    }

    printf ("Finished.\n");
}
#endif

static Landmark_warp*
load_input_files (args_info_landmark_warp *args_info)
{
    Landmark_warp *lw = 0;

    /* Load the landmark data */
    if (args_info->input_xform_arg) {
	lw = landmark_warp_load_xform (args_info->input_xform_arg);
	if (!lw) {
	    print_and_exit ("Error, landmarks were not loaded successfully.\n");
	}
    }
    else if (args_info->fixed_landmarks_arg && args_info->moving_landmarks_arg)
    {
	lw = landmark_warp_load_pointsets (
	    args_info->fixed_landmarks_arg, 
	    args_info->moving_landmarks_arg);
	if (!lw) {
	    print_and_exit ("Error, landmarks were not loaded successfully.\n");
	}
    } else {
	print_and_exit (
	    "Error.  Input landmarks must be specified using either the "
	    "--input-xform option\nor the --fixed-landmarks and "
	    "--moving-landmarks option.\n");
    }

    /* Load the input image */
    lw->m_input_img = plm_image_load_native (args_info->input_image_arg);
    if (!lw->m_input_img) {
	print_and_exit ("Error reading moving file: %s\n", 
	    (const char*) args_info->input_image_arg);
    }
    return lw;
}

static void
do_landmark_warp (args_info_landmark_warp *args_info)
{
    Landmark_warp *lw;

    lw = load_input_files (args_info);

    switch (args_info->algorithm_arg) {
    case algorithm_arg_itk:
	do_landmark_warp_itk_tps (lw);
    case algorithm_arg_nsh:
    case algorithm_arg_gcs:
    default:
	break;
    }
}

static void
check_arguments (args_info_landmark_warp *args_info)
{
    /* Nothing to check? */
}

int
main (int argc, char *argv[])
{
    GGO (landmark_warp, args_info);
    check_arguments (&args_info);

    do_landmark_warp (&args_info);

    GGO_FREE (landmark_warp, args_info);

    return 0;
}
