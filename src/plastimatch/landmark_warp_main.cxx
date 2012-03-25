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
#include "plm_clp.h"
#include "pointset.h"
#include "print_and_exit.h"
#include "pstring.h"
#include "rbf_gauss.h"
#include "rbf_wendland.h"

class Landmark_warp_main_parms {
public:
    Pstring fixed_lm_fn;
    Pstring moving_lm_fn;
    Pstring input_vf_fn;
    Pstring input_img_fn;
    Pstring output_img_fn;
    Pstring output_vf_fn;
    Pstring output_lm_fn;

    Pstring fixed_img_fn;
    bool have_dim;
    plm_long dim[3];
    bool have_origin;
    float origin[3];
    bool have_spacing;
    float spacing[3];
    bool have_direction_cosines;
    float direction_cosines[9];

    Pstring algorithm;

    Landmark_warp lw;

public:
    Landmark_warp_main_parms () {
	have_dim = 0;
	have_origin = 0;
	have_spacing = 0;
	have_direction_cosines = 0;
    }
};

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

static void
load_input_files (Landmark_warp_main_parms *parms)
{
    Landmark_warp *lw = &parms->lw;

    /* Load the pointsets */
    lw->load_pointsets (parms->fixed_lm_fn, parms->moving_lm_fn);
    if (!lw) {
	print_and_exit ("Error, landmarks were not loaded successfully.\n");
    }

    /* Load the input image */
    if (bstring_not_empty (parms->input_img_fn)) {
	lw->m_input_img = plm_image_load_native (parms->input_img_fn);
	if (!lw->m_input_img) {
	    print_and_exit ("Error reading input image: %s\n", 
		(const char*) parms->input_img_fn);
	}
	/* Default geometry, if unknown, comes from moving image */
	lw->m_pih.set_from_plm_image (lw->m_input_img);
    }

    /* Set the output geometry.  
       Note: --offset, --spacing, and --dim get priority over --fixed. 
       Therefore, if these options are completely specified, we don't need 
       to load the fixed image.
    */
    if (!parms->have_dim || !parms->have_origin 
	|| !parms->have_spacing || !parms->have_direction_cosines)
    {
	if (bstring_not_empty (parms->fixed_img_fn)) {
	    Plm_image *pli = plm_image_load_native (parms->fixed_img_fn);
	    if (!pli) {
		print_and_exit ("Error loading fixed image: %s\n",
		    (const char*) parms->fixed_img_fn);
	    }
	    lw->m_pih.set_from_plm_image (pli);
	    delete pli;
	}
    }
    if (parms->have_dim) {
	lw->m_pih.set_dim (parms->dim);
    }
    if (parms->have_origin) {
	lw->m_pih.set_origin (parms->origin);
    }
    if (parms->have_spacing) {
	lw->m_pih.set_spacing (parms->spacing);
    }
//	if (parms->have_direction_cosines) {
//	lw->m_pih.set_direction_cosines (parms->direction_cosines);
//    }
}

static void
save_output_files (Landmark_warp_main_parms *parms)
{
    Landmark_warp *lw = &parms->lw;

    /* GCS FIX: float output only, and no dicom. */
    if (lw->m_warped_img && bstring_not_empty (parms->output_img_fn)) {
	lw->m_warped_img->save_image (parms->output_img_fn);
    }
    if (lw->m_vf && bstring_not_empty (parms->output_vf_fn)) {
	xform_save (lw->m_vf, (const char*) parms->output_vf_fn);
    }
    if (lw->m_vf && lw->m_warped_landmarks 
	&& bstring_not_empty (parms->output_lm_fn))
    {
	
	if ( lw->num_clusters > 0 ) {
	    // if clustering required, save each cluster to a separate fcsv
	    for( int ii = 0; ii<lw->num_clusters; ii++) {
		// remove .fcsv extension, insert cluster id, add back extension
		char fn_out[1024], clust_name[1024];
		strcpy(fn_out, parms->output_lm_fn );
		char *ext_pos = strstr(fn_out, ".fcsv");
		fn_out[ext_pos-fn_out]=0;
		sprintf(clust_name, "_cl_%d", ii);	
		strcat(fn_out, clust_name);
		strcat(fn_out, ".fcsv");		
		
		//pointset_save_fcsv_by_cluster(lw->m_warped_landmarks, 
		//	lw->cluster_id, ii,  fn_out);
		
		//write FIXED landmarks to check for clustering issues
		pointset_save_fcsv_by_cluster(lw->m_fixed_landmarks, 
			lw->cluster_id, ii,  fn_out);
		}
	    }
	    else pointset_save (lw->m_warped_landmarks, parms->output_lm_fn);
    }
}

/*
  calculate voxel positions of landmarks given offset and pix_spacing
  output: landvox
  input: landmarks_mm and offset/spacing/dim
  NOTE: ROTATION IS NOT SUPPORTED! direction_cosines assumed to be 100 010 001.
*/
static void 
landmark_convert_mm_to_voxel (
    int *landvox, 
    Raw_pointset *landmarks_mm, 
    float *offset, 
    float *pix_spacing,
    plm_long *dim,
    const float *direction_cosines)
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
    plm_long ri, rj, rk;
    plm_long fi, fj, fk, fv;
    plm_long mi, mj, mk;
    float fx, fy, fz;
    float mx, my, mz;
    int i, d, lidx;
    float dd, *vf, dxyz[3], *dd_min;
    
    int num_landmarks;
    int *landvox_mov, *landvox_fix, *landvox_warp;
    float *warped_landmarks;
    float *landmark_dxyz;
    Volume *vector_field;
    Volume *moving;
    plm_long fixed_dim[3];
    float fixed_spacing[3], fixed_offset[3], fixed_direction_cosines[9];

    num_landmarks = lw->m_fixed_landmarks->num_points;

    landvox_mov  = (int *)malloc( 3*num_landmarks * sizeof(int));
    landvox_fix  = (int *)malloc( 3*num_landmarks * sizeof(int));
    landvox_warp = (int *)malloc( 3*num_landmarks * sizeof(int));
    landmark_dxyz = (float *)malloc( 3*num_landmarks * sizeof(float));
    warped_landmarks = (float *)malloc( 3*num_landmarks * sizeof(float));

    vector_field = lw->m_vf->get_gpuit_vf();
    moving = lw->m_input_img->gpuit_float();

    /* fixed dimensions set come from lw->m_pih */
    lw->m_pih.get_dim (fixed_dim);
    lw->m_pih.get_spacing (fixed_spacing);
    lw->m_pih.get_origin (fixed_offset);
    lw->m_pih.get_direction_cosines (fixed_direction_cosines);

    if (vector_field->pix_type != PT_VF_FLOAT_INTERLEAVED)
	print_and_exit ("Sorry, this type of vector field is not supported in landmarks_warp\n");	
    vf = (float *)vector_field->img;

    /* fill in landvox'es */
    landmark_convert_mm_to_voxel (landvox_fix, lw->m_fixed_landmarks, 
	fixed_offset, fixed_spacing, fixed_dim, fixed_direction_cosines);
    landmark_convert_mm_to_voxel (landvox_mov, lw->m_moving_landmarks, 
	moving->offset, moving->spacing, moving->dim, 
	moving->direction_cosines);
    
    dd_min = (float *)malloc( num_landmarks * sizeof(float));
    for (d=0;d<num_landmarks;d++) dd_min[d] = 1e20F; //a very large number

    /* roi_offset and roi_dim are not used here */

    for (rk = 0, fk = 0; rk < fixed_dim[2]; rk++, fk++) {
	fz = fixed_offset[2] + fixed_spacing[2] * fk;
	for (rj = 0, fj = 0; rj < fixed_dim[1]; rj++, fj++) {
	    fy = fixed_offset[1] + fixed_spacing[1] * fj;
	    for (ri = 0, fi = 0; ri < fixed_dim[0]; ri++, fi++) {
		fx = fixed_offset[0] + fixed_spacing[0] * fi;

		fv = fk * vector_field->dim[0] * vector_field->dim[1] 
		    + fj * vector_field->dim[0] +fi ;

		for (d=0;d<3;d++) dxyz[d] = vf[3*fv+d];

		/* Find correspondence in moving image */
		mx = fx + dxyz[0];
		mi = ROUND_INT ((mx - moving->offset[0]) / moving->spacing[0]);
		if (mi < 0 || mi >= moving->dim[0]) continue;
		my = fy + dxyz[1];
		mj = ROUND_INT ((my - moving->offset[1]) / moving->spacing[1]);
		if (mj < 0 || mj >= moving->dim[1]) continue;
		mz = fz + dxyz[2];
		mk = ROUND_INT ((mz - moving->offset[2]) / moving->spacing[2]);
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

    for (i = 0; i < num_landmarks; i++) {
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
			- fixed_offset[d]) / fixed_spacing[d]);
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
    for (lidx = 0; lidx < num_landmarks; lidx++)
    {
	fx=0;
	for (d = 0; d < 3; d++) { 
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
do_landmark_warp (Landmark_warp_main_parms *parms)
{
    Landmark_warp *lw = &parms->lw;

    load_input_files (parms);

    if (parms->algorithm == "tps") {
	do_landmark_warp_itk_tps (lw);
    }
    else if (parms->algorithm == "gauss") {
	do_landmark_warp_gauss (lw);
    }
    else if (parms->algorithm == "wendland") {
	do_landmark_warp_wendland (lw);
    }

    if (bstring_not_empty (parms->output_lm_fn)) {
	lw->m_warped_landmarks = pointset_create ();
	calculate_warped_landmarks (lw);
    }

    save_output_files (parms);
}

static void
usage_fn (dlib::Plm_clp* parser, int argc, char *argv[])
{
    std::cout << "Usage: landmark_warp [options]\n";
    parser->print_options (std::cout);
    std::cout << std::endl;
}

static void
parse_fn (
    Landmark_warp_main_parms *parms, 
    dlib::Plm_clp *parser, 
    int argc, 
    char* argv[]
)
{
    Landmark_warp *lw = &parms->lw;

    /* Basic options */
    parser->add_long_option ("h", "help", "Display this help message");
    parser->add_long_option ("f", "fixed-landmarks", 
	"Input fixed landmarks", 1, "");
    parser->add_long_option ("m", "moving-landmarks",
	"Output moving landmarks", 1, "");
    parser->add_long_option ("v", "input-vf",
	"Input vector field (applied prior to landmark warping)", 1, "");
    parser->add_long_option ("I", "input-image",
	"Input image to warp", 1, "");
    parser->add_long_option ("O", "output-image",
	"Output warped image", 1, "");
    parser->add_long_option ("V", "output-vf", 
	"Output vector field", 1, "");
    parser->add_long_option ("L", "output-landmarks", 
	"Output warped landmarks", 1, "");

    /* Output geometry options */
    parser->add_long_option ("", "origin", 
	"Location of first image voxel in mm \"x y z\"", 1, "");
    parser->add_long_option ("", "spacing", 
	"Voxel spacing in mm \"x [y z]\"", 1, "");
    parser->add_long_option ("", "dim", 
	"Size of output image in voxels \"x [y z]\"", 1, "");
    parser->add_long_option ("F", "fixed", 
	"Fixed image (match output size to this image)", 1, "");

    /* Algorithm options */
    parser->add_long_option ("a", "algorithm",
	"RBF warping algorithm {tps,gauss,wendland}", 1, "gauss");
    parser->add_long_option ("r", "radius",
	"Radius of radial basis function (in mm)", 1, "50.0");
    parser->add_long_option ("Y", "stiffness",
	"Young modulus (default = 0.0)", 1, "0.0");
    parser->add_long_option ("d", "default-value",
	"Value to set for pixels with unknown value", 1, "-1000");
    parser->add_long_option ("N", "numclusters",
	"Number of clusters of landmarks", 1, "0");

    /* Parse the command line arguments */
    parser->parse (argc,argv);

    /* Check if the -h option was given */
    parser->check_help ();

    /* Check that required inputs were given */
    parser->check_required ("fixed-landmarks");
    parser->check_required ("moving-landmarks");

    /* Copy values into output struct */
    parms->fixed_lm_fn = parser->get_string("fixed-landmarks").c_str();
    parms->moving_lm_fn = parser->get_string("moving-landmarks").c_str();
    parms->input_vf_fn = parser->get_string("input-vf").c_str();
    parms->input_img_fn = parser->get_string("input-image").c_str();
    parms->output_img_fn = parser->get_string("output-image").c_str();
    parms->output_vf_fn = parser->get_string("output-vf").c_str();
    parms->output_lm_fn = parser->get_string("output-landmarks").c_str();

    if (parser->option ("origin")) {
	parms->have_origin = 1;
	parser->assign_float13 (parms->origin, "origin");
    }
    if (parser->option ("spacing")) {
	parms->have_spacing = 1;
	parser->assign_float13 (parms->spacing, "spacing");
    }
    if (parser->option ("dim")) {
	parms->have_dim = 1;
	parser->assign_plm_long_13 (parms->dim, "dim");
    }
    parms->fixed_img_fn = parser->get_string("fixed").c_str();

    parms->algorithm = parser->get_string("algorithm").c_str();
    lw->rbf_radius = parser->get_float("radius");
    lw->young_modulus = parser->get_float("stiffness");
    lw->default_val = parser->get_float("default-value");
    lw->num_clusters = parser->get_float("numclusters");
}

int
main (int argc, char *argv[])
{
    Landmark_warp_main_parms parms;

    plm_clp_parse (&parms, &parse_fn, &usage_fn, argc, argv);
    do_landmark_warp (&parms);

    return 0;
}
