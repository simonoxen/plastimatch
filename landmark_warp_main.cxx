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
#include "itk_tps.h"
#include "math_util.h"
#include "mha_io.h"
#include "landmark_warp.h"
#include "landmark_warp_args.h"
#include "landmark_warp_ggo.h"
#include "plm_ggo.h"
#include "pointset.h"
#include "print_and_exit.h"
#include "rbf_gauss.h"
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
    itk_tps_warp (lw);
}

static void
do_landmark_warp_gcs (Landmark_warp *lw)
{
    rbf_gcs_warp (lw);
}

static void
do_landmark_warp_nsh (Landmark_warp *lw)
{
    rbf_gauss_warp (lw);
}

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

    /* Set the output geometry.  
       Note: --offset, --spacing, and --dim get priority over --fixed. */
    if (!args_info->origin_arg 
	|| !args_info->spacing_arg 
	|| !args_info->dim_arg) 
    {
	if (args_info->fixed_arg) {
	    Plm_image *pli = plm_image_load_native (args_info->fixed_arg);
	    if (!pli) {
		print_and_exit ("Error loading fixed image: %s\n",
		    args_info->fixed_arg);
	    }
	    lw->m_pih.set_from_plm_image (pli);
	    delete pli;
	} else {
	    lw->m_pih.set_from_plm_image (lw->m_input_img);
	}
    }
    if (args_info->origin_arg) {
	int rc;
	float f[3];
	rc = sscanf (args_info->origin_arg, "%f %f %f", &f[0], &f[1], &f[2]);
	if (rc != 3) {
	    print_and_exit ("Error parsing origin: %s\n",
		args_info->origin_arg);
	}
	lw->m_pih.set_origin (f);
    }
    if (args_info->spacing_arg) {
	int rc;
	float f[3];
	rc = sscanf (args_info->spacing_arg, "%f %f %f", &f[0], &f[1], &f[2]);
	if (rc != 3) {
	    print_and_exit ("Error parsing spacing: %s\n",
		args_info->spacing_arg);
	}
	lw->m_pih.set_spacing (f);
    }
    if (args_info->dim_arg) {
	int rc;
	int d[3];
	rc = sscanf (args_info->dim_arg, "%d %d %d", &d[0], &d[1], &d[2]);
	if (rc != 3) {
	    print_and_exit ("Error parsing dim: %s\n",
		args_info->dim_arg);
	}
	lw->m_pih.set_dim (d);
    }

    return lw;
}

static void
save_output_files (Landmark_warp *lw, args_info_landmark_warp *args_info)
{
    /* GCS FIX: float output only, and no dicom. */
    if (lw->m_warped_img && args_info->output_image_arg) {
	lw->m_warped_img->save_image (args_info->output_image_arg);
    }
    if (lw->m_vf && args_info->output_vf_arg) {
	xform_save (lw->m_vf, args_info->output_vf_arg);
    }
}

static void
do_landmark_warp (args_info_landmark_warp *args_info)
{
    Landmark_warp *lw;

    lw = load_input_files (args_info);

    switch (args_info->algorithm_arg) {
    case algorithm_arg_itk:
	do_landmark_warp_itk_tps (lw);
	break;
    case algorithm_arg_nsh:
	break;
    case algorithm_arg_gcs:
	do_landmark_warp_gcs (lw);
	break;
    default:
	break;
    }

    save_output_files (lw, args_info);
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
