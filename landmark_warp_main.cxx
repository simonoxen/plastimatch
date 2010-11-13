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
load_input_files (Landmark_warp_args *parms)
{
    Landmark_warp *lw = 0;

    /* Load the landmark data */
    if (bstring_not_empty (parms->input_xform_fn)) {
	lw = landmark_warp_load_xform ((const char*) parms->input_xform_fn);
    }
    else if (bstring_not_empty (parms->input_fixed_landmarks_fn) 
	&& bstring_not_empty (parms->input_moving_landmarks_fn))
    {
	lw = landmark_warp_load_pointsets (
	    (const char*) parms->input_fixed_landmarks_fn, 
	    (const char*) parms->input_moving_landmarks_fn);
    }
    if (!lw) {
	print_and_exit ("Error, landmarks were not loaded successfully.\n");
    }

    /* Load the input image */
    lw->m_img = plm_image_load_native (parms->input_moving_image_fn);
    if (!lw->m_img) {
	print_and_exit ("Error reading moving file: %s\n", 
	    (const char*) parms->input_moving_image_fn);
    }
    return lw;
}

static void
do_landmark_warp (Landmark_warp_args *parms)
{
    Landmark_warp *lw;

    lw = load_input_files (parms);

    switch (parms->m_algorithm) {
    case LANDMARK_WARP_ALGORITHM_ITK_TPS:
	break;
    case LANDMARK_WARP_ALGORITHM_RBF_NSH:
	break;
    case LANDMARK_WARP_ALGORITHM_RBF_GCS:
    default:
	break;
    }

}

int
main (int argc, char *argv[])
{
    Landmark_warp_args parms;

    GGO (landmark_warp, args_info);

    parms.parse_args (argc, argv);
    do_landmark_warp (&parms);
    return 0;
}
