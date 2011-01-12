/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plm_config.h"
#include <stdio.h>
#include <iostream>
#include <vector>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#include "bstring_util.h"
#include "itk_tps.h"
#include "math_util.h"
#include "mha_io.h"
#include "landmark_warp.h"
//#include "landmark_warp_args.h"
#include "landmark_warp_ggo.h"
//#include "plm_ggo.h"
#include "pointset.h"
#include "print_and_exit.h"
#include "rbf_gauss.h"
#include "rbf_gcs.h"

// this .h is generated from ...landwarp.xml file by GenerateCLP in Slicer3-build
#include "plastimatch-slicer-landwarpCLP.h"

/* 
NSh: Code below, up until main() is verbatim landmark_warp_main.cxx, 
unless marked debug or NSh
*/

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

//Load the input image 
    lw->m_input_img = plm_image_load_native (args_info->input_image_arg);
    if (!lw->m_input_img) 
    {
	print_and_exit ("Error reading moving file: %s\n", (const char*) args_info->input_image_arg);
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

    lw->rbf_radius = args_info->radius_arg;
    lw->young_modulus = args_info->stiffness_arg;

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
do_landmark_warp (Landmark_warp *lw, const char *algorithm)
{
    if (!strcmp (algorithm, "tps")) {
	do_landmark_warp_itk_tps (lw);
    }
    else if (!strcmp (algorithm, "gauss")) {
	do_landmark_warp_nsh (lw);
    }
    else {
	do_landmark_warp_gcs (lw);
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
    //PARSE_ARGS comes from ...CLP.h
    PARSE_ARGS;

    Landmark_warp *lw = landmark_warp_create ();

    lw->m_input_img = plm_image_load_native (
	plmslc_landwarp_moving_volume.c_str());
    Plm_image *tmp = plm_image_load_native (
	plmslc_landwarp_fixed_volume.c_str());
    lw->m_pih.set_from_plm_image (tmp);
    lw->default_val=plmslc_landwarp_default_value;
    lw->rbf_radius=plmslc_landwarp_rbf_radius;
    lw->young_modulus=plmslc_landwarp_stiffness;

    delete tmp;

    unsigned long num_fiducials = plmslc_landwarp_fixed_fiducials.size();
    if (plmslc_landwarp_moving_fiducials.size() < num_fiducials) {
	num_fiducials = plmslc_landwarp_moving_fiducials.size();
    }

    /* NSh: pointset_load_fcsv assumes RAS, as does Slicer.
       For some reason, pointset_load_txt assumes LPS.
       Thus, we write out Slicer-style .fcsv
    */

    Pointset *fix_ps = pointset_create ();
    Pointset *mov_ps = pointset_create ();

    for (unsigned long i = 0; i < num_fiducials; i++) {
	
	float lm_fix[3] = { plmslc_landwarp_fixed_fiducials[i][0],  plmslc_landwarp_fixed_fiducials[i][1],  plmslc_landwarp_fixed_fiducials[i][2]};
	pointset_add_point (fix_ps, lm_fix);
    
	float lm_mov[3] = { plmslc_landwarp_moving_fiducials[i][0],  plmslc_landwarp_moving_fiducials[i][1],  plmslc_landwarp_moving_fiducials[i][2]};
	pointset_add_point (mov_ps, lm_mov);
    }

    lw->m_fixed_landmarks = fix_ps;
    lw->m_moving_landmarks = mov_ps;

    do_landmark_warp (lw, plmslc_landwarp_rbf_type.c_str());
    if (lw->m_warped_img && plmslc_landwarp_warped_volume != "None") {
	lw->m_warped_img->save_image (plmslc_landwarp_warped_volume.c_str());
    }
    return EXIT_SUCCESS;
}
