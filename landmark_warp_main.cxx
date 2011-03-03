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
#include "rbf_wendland.h"

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
do_landmark_warp_wendland (Landmark_warp *lw)
{
    rbf_wendland_warp (lw);
}

static void
do_landmark_warp_gauss (Landmark_warp *lw)
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

    lw->rbf_radius = args_info->radius_arg;
    lw->young_modulus = args_info->stiffness_arg;
	lw->num_clusters = args_info->numclusters_arg;
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
    if (lw->m_vf && lw->m_warped_landmarks && args_info->output_landmarks_arg) {
	pointset_save (lw->m_warped_landmarks, args_info->output_landmarks_arg);
    }

}

/*
calculate voxel positions of landmarks given offset and pix_spacing
output: landvox
input: landmarks_mm and offset/spacing/dim
NOTE: ROTATION IS NOT SUPPORTED! direction_cosines assumed to be 100 010 001.
*/
static void 
landmark_convert_mm_to_voxel(
int *landvox, 
Pointset *landmarks_mm, 
float *offset, 
float *pix_spacing,
int *dim )
{
    for (int i = 0; i < landmarks_mm->num_points; i++) {
	for (int d = 0; d < 3; d++) {
	    landvox[i*3 + d] = ROUND_INT (
		( landmarks_mm->points[3*i + d]
		    - offset[d]) / pix_spacing[d]);
	    if (landvox[i*3 + d] < 0 
		|| landvox[i*3 + d] >= dim[d])
	    {
		print_and_exit (
		    "Error, landmark %d outside of image for dim %d.\n"
		    "Location in vox = %d\n"
		    "Image boundary in vox = (%d %d)\n",
		    i, d, landvox[i*3 + d], 0, dim[d]-1);
	    }
	}
    }


}

static void 
calculate_warped_landmarks( Landmark_warp *lw )
/*
Moves moving landmarks according to the current vector field.
Output goes into lw->m_warped_landmarks
LW = warped landmark
We must solve LW + u(LW) = LM to get new LW, corresponding to current vector field.

Adapted from bspline_landmarks_warp(...) in bspline_landmarks.c to use Landmark_warp

*/
{
    int ri, rj, rk;
    int fi, fj, fk;
    int mi, mj, mk;
    float fx, fy, fz;
    float mx, my, mz;
    int i,d,fv, lidx;
    float dd, *vf, dxyz[3], *dd_min;
    
    int num_landmarks;
    int *landvox_mov, *landvox_fix, *landvox_warp;
    float *warped_landmarks;
    float *landmark_dxyz;
    Volume *vector_field;
    Volume *moving;
    int fixed_dim[3];
    float fixed_pix_spacing[3], fixed_offset[3];

    num_landmarks = lw->m_fixed_landmarks->num_points;

    landvox_mov  = (int *)malloc( 3*num_landmarks * sizeof(int));
    landvox_fix  = (int *)malloc( 3*num_landmarks * sizeof(int));
    landvox_warp = (int *)malloc( 3*num_landmarks * sizeof(int));
    landmark_dxyz = (float *)malloc( 3*num_landmarks * sizeof(float));
    warped_landmarks = (float *)malloc( 3*num_landmarks * sizeof(float));

    vector_field = lw->m_vf->get_gpuit_vf();
    moving = lw->m_input_img->gpuit_float();

    /* fixed dimensions set come from lw->m_pih */
    lw->m_pih.get_dim( fixed_dim);
    lw->m_pih.get_spacing( fixed_pix_spacing );
    lw->m_pih.get_origin( fixed_offset );

    if (vector_field->pix_type != PT_VF_FLOAT_INTERLEAVED)
	print_and_exit ("Sorry, this type of vector field is not supported in landmarks_warp\n");	
    vf = (float *)vector_field->img;

    /* fill in landvox'es */
    landmark_convert_mm_to_voxel( landvox_fix, lw->m_fixed_landmarks, 
		fixed_offset, fixed_pix_spacing, fixed_dim );
    landmark_convert_mm_to_voxel( landvox_mov, lw->m_moving_landmarks, 
		moving->offset, moving->pix_spacing, moving->dim );
    
    dd_min = (float *)malloc( num_landmarks * sizeof(float));
    for (d=0;d<num_landmarks;d++) dd_min[d] = 1e20F; //a very large number

    /* roi_offset and roi_dim are not used here */

    for (rk = 0, fk = 0; rk < fixed_dim[2]; rk++, fk++) {
	fz = fixed_offset[2] + fixed_pix_spacing[2] * fk;
	for (rj = 0, fj = 0; rj < fixed_dim[1]; rj++, fj++) {
	    fy = fixed_offset[1] + fixed_pix_spacing[1] * fj;
	    for (ri = 0, fi = 0; ri < fixed_dim[0]; ri++, fi++) {
		fx = fixed_offset[0] + fixed_pix_spacing[0] * fi;

		fv = fk * vector_field->dim[0] * vector_field->dim[1] 
		    + fj * vector_field->dim[0] +fi ;

		for (d=0;d<3;d++) dxyz[d] = vf[3*fv+d];

		/* Find correspondence in moving image */
		mx = fx + dxyz[0];
		mi = ROUND_INT ((mx - moving->offset[0]) / moving->pix_spacing[0]);
		if (mi < 0 || mi >= moving->dim[0]) continue;
		my = fy + dxyz[1];
		mj = ROUND_INT ((my - moving->offset[1]) / moving->pix_spacing[1]);
		if (mj < 0 || mj >= moving->dim[1]) continue;
		mz = fz + dxyz[2];
		mk = ROUND_INT ((mz - moving->offset[2]) / moving->pix_spacing[2]);
		if (mk < 0 || mk >= moving->dim[2]) continue;

		//saving vector field in a voxel which is the closest to landvox_mov
		//after being displaced by the vector field
		for (lidx = 0; lidx < num_landmarks; lidx++) {
		    dd = (mi - landvox_mov[lidx*3+0]) * (mi - landvox_mov[lidx*3+0])
			+(mj - landvox_mov[lidx*3+1]) * (mj - landvox_mov[lidx*3+1])
			+(mk - landvox_mov[lidx*3+2]) * (mk - landvox_mov[lidx*3+2]);
		    if (dd < dd_min[lidx]) { 
			dd_min[lidx]=dd;   
			for (d=0;d<3;d++) {
			    landmark_dxyz[3*lidx+d] = vf[3*fv+d];
			}
		    }
		} 
	    }
	}
    }

    for (i=0;i<num_landmarks;i++)  {
	for (d=0; d<3; d++) {
	    warped_landmarks[3*i+d]
		= lw->m_moving_landmarks->points[3*i+d]
		- landmark_dxyz[3*i+d];
	}
    }

    /* calculate voxel positions of warped landmarks  */
    for (lidx = 0; lidx < num_landmarks; lidx++) {
	for (d = 0; d < 3; d++) {
	    landvox_warp[lidx*3 + d] 
		= ROUND_INT ((warped_landmarks[lidx*3 + d] 
			- fixed_offset[d]) / fixed_pix_spacing[d]);
	    if (landvox_warp[lidx*3 + d] < 0 
		|| landvox_warp[lidx*3 + d] >= fixed_dim[d])
	    {
		print_and_exit (
		    "Error, warped landmark %d outside of fixed image for dim %d.\n"
		    "Location in vox = %d\n"
		    "Image boundary in vox = (%d %d)\n",
		    lidx, d, landvox_warp[lidx*3 + d], 0, fixed_dim[d]-1);
	    }
	} 
    pointset_add_point_noadjust (lw->m_warped_landmarks, warped_landmarks+3*lidx);
    }

//debug only
fy = 0;
for(lidx=0;lidx<num_landmarks;lidx++)
{
fx=0;
for(d=0;d<3;d++) { 
    fz = (lw->m_fixed_landmarks->points[3*lidx+d] - lw->m_warped_landmarks->points[3*lidx+d] );
    fx += fz*fz;
}
printf("landmark %3d err %f mm\n", lidx, sqrt(fx));
fy+=fx;
}
printf("landmark RMS err %f mm\n", sqrt(fy/num_landmarks));
// end debug only

    free (dd_min);
    free (landvox_mov);
    free (landvox_warp);
    free (landvox_fix);
    free (landmark_dxyz);
    free (warped_landmarks);
}


static void
do_landmark_warp (args_info_landmark_warp *args_info)
{
    Landmark_warp *lw;

    lw = load_input_files (args_info);

    switch (args_info->algorithm_arg) {
    case algorithm_arg_tps:
	do_landmark_warp_itk_tps (lw);
	break;
    case algorithm_arg_gauss:
	do_landmark_warp_gauss (lw);
	break;
    case algorithm_arg_wendland:
	do_landmark_warp_wendland (lw);
	break;
    default:
	break;
    }

    if ( args_info->output_landmarks_arg ) {  
	lw->m_warped_landmarks = pointset_create ();
	calculate_warped_landmarks( lw );
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
    GGO (landmark_warp, args_info, 0);
    check_arguments (&args_info);

    do_landmark_warp (&args_info);

    GGO_FREE (landmark_warp, args_info, 0);
    return 0;
}
