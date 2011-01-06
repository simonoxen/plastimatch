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
//#include "landmark_warp_ggo.h"
//#include "plm_ggo.h"
#include "pointset.h"
#include "print_and_exit.h"
#include "rbf_gauss.h"
#include "rbf_gcs.h"

// this .h is generated from ...landwarp.xml file by GenerateCLP in Slicer3-build
#include "plastimatch-slicer-landwarpCLP.h"

FILE *fpdebug;

// NSh: copy structure definitions to reuse GGO dependent code 
// from landmark_warp.cxx with TCLAP/GenerateCLP required by Slicer
enum enum_algorithm { algorithm_arg_tps = 0 , algorithm_arg_gauss, algorithm_arg_cone };

struct args_info_landmark_warp
{
  const char *help_help; /**< @brief Print help and exit help description.  */
  const char *full_help_help; /**< @brief Print help, including hidden options, and exit help description.  */
  const char *version_help; /**< @brief Print version and exit help description.  */
  char * fixed_landmarks_arg;	/**< @brief Input fixed landmarks.  */
  char * fixed_landmarks_orig;	/**< @brief Input fixed landmarks original value given at command line.  */
  const char *fixed_landmarks_help; /**< @brief Input fixed landmarks help description.  */
  char * moving_landmarks_arg;	/**< @brief Input moving landmarks.  */
  char * moving_landmarks_orig;	/**< @brief Input moving landmarks original value given at command line.  */
  const char *moving_landmarks_help; /**< @brief Input moving landmarks help description.  */
  char * input_xform_arg;	/**< @brief Input landmark xform.  */
  char * input_xform_orig;	/**< @brief Input landmark xform original value given at command line.  */
  const char *input_xform_help; /**< @brief Input landmark xform help description.  */
  char * input_image_arg;	/**< @brief Input image to warp.  */
  char * input_image_orig;	/**< @brief Input image to warp original value given at command line.  */
  const char *input_image_help; /**< @brief Input image to warp help description.  */
  char * output_image_arg;	/**< @brief Output warped image.  */
  char * output_image_orig;	/**< @brief Output warped image original value given at command line.  */
  const char *output_image_help; /**< @brief Output warped image help description.  */
  char * output_vf_arg;	/**< @brief Output vector field.  */
  char * output_vf_orig;	/**< @brief Output vector field original value given at command line.  */
  const char *output_vf_help; /**< @brief Output vector field help description.  */
  char * origin_arg;	/**< @brief Output image offset.  */
  char * origin_orig;	/**< @brief Output image offset original value given at command line.  */
  const char *origin_help; /**< @brief Output image offset help description.  */
  char * spacing_arg;	/**< @brief Output image spacing.  */
  char * spacing_orig;	/**< @brief Output image spacing original value given at command line.  */
  const char *spacing_help; /**< @brief Output image spacing help description.  */
  char * dim_arg;	/**< @brief Output image dimension.  */
  char * dim_orig;	/**< @brief Output image dimension original value given at command line.  */
  const char *dim_help; /**< @brief Output image dimension help description.  */
  char * fixed_arg;	/**< @brief Fixed image (match output size to this image).  */
  char * fixed_orig;	/**< @brief Fixed image (match output size to this image) original value given at command line.  */
  const char *fixed_help; /**< @brief Fixed image (match output size to this image) help description.  */
  enum enum_algorithm algorithm_arg;	/**< @brief RBF warping algorithm  (default='cone').  */
  char * algorithm_orig;	/**< @brief RBF warping algorithm  original value given at command line.  */
  const char *algorithm_help; /**< @brief RBF warping algorithm  help description.  */
  float radius_arg;	/**< @brief Radius of radial basis function (default='50.0').  */
  char * radius_orig;	/**< @brief Radius of radial basis function original value given at command line.  */
  const char *radius_help; /**< @brief Radius of radial basis function help description.  */
  float stiffness_arg;	/**< @brief Young modulus (default='1.0').  */
  char * stiffness_orig;	/**< @brief Young modulus original value given at command line.  */
  const char *stiffness_help; /**< @brief Young modulus help description.  */
  float default_value_arg;	/**< @brief Value to set for pixels with unknown value (default='0.0').  */
  char * default_value_orig;	/**< @brief Value to set for pixels with unknown value original value given at command line.  */
  const char *default_value_help; /**< @brief Value to set for pixels with unknown value help description.  */
  char * config_arg;	/**< @brief Config file.  */
  char * config_orig;	/**< @brief Config file original value given at command line.  */
  const char *config_help; /**< @brief Config file help description.  */
  
  unsigned int help_given ;	/**< @brief Whether help was given.  */
  unsigned int full_help_given ;	/**< @brief Whether full-help was given.  */
  unsigned int version_given ;	/**< @brief Whether version was given.  */
  unsigned int fixed_landmarks_given ;	/**< @brief Whether fixed-landmarks was given.  */
  unsigned int moving_landmarks_given ;	/**< @brief Whether moving-landmarks was given.  */
  unsigned int input_xform_given ;	/**< @brief Whether input-xform was given.  */
  unsigned int input_image_given ;	/**< @brief Whether input-image was given.  */
  unsigned int output_image_given ;	/**< @brief Whether output-image was given.  */
  unsigned int output_vf_given ;	/**< @brief Whether output-vf was given.  */
  unsigned int origin_given ;	/**< @brief Whether origin was given.  */
  unsigned int spacing_given ;	/**< @brief Whether spacing was given.  */
  unsigned int dim_given ;	/**< @brief Whether dim was given.  */
  unsigned int fixed_given ;	/**< @brief Whether fixed was given.  */
  unsigned int algorithm_given ;	/**< @brief Whether algorithm was given.  */
  unsigned int radius_given ;	/**< @brief Whether radius was given.  */
  unsigned int stiffness_given ;	/**< @brief Whether stiffness was given.  */
  unsigned int default_value_given ;	/**< @brief Whether default-value was given.  */
  unsigned int config_given ;	/**< @brief Whether config was given.  */

  char **inputs ; /**< @brief unamed options (options without names) */
  unsigned inputs_num ; /**< @brief unamed options number */
} ;

/* 
NSh: Code below, up until main() is verbatim landmark_warp.cxx, 
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
fprintf(fpdebug,"enter gauss warp\n"); fflush(fpdebug);

	rbf_gauss_warp (lw);
fprintf(fpdebug,"exit gauss warp\n"); fflush(fpdebug);

}

static Landmark_warp*
load_input_files (args_info_landmark_warp *args_info)
{
    Landmark_warp *lw = 0;

    /* Load the landmark data */
    if (args_info->input_xform_arg) {
	lw = landmark_warp_load_xform (args_info->input_xform_arg);
	if (!lw) {
		fprintf(fpdebug,"error at load_xform\n"); fflush(fpdebug);
		print_and_exit ("Error, landmarks were not loaded successfully.\n");
	}
    }
    else if (args_info->fixed_landmarks_arg && args_info->moving_landmarks_arg)
    {
	lw = landmark_warp_load_pointsets (
	    args_info->fixed_landmarks_arg, 
	    args_info->moving_landmarks_arg);
	if (!lw) {
		fprintf(fpdebug,"error at load_pointsets\n"); fflush(fpdebug);
		print_and_exit ("Error, landmarks were not loaded successfully.\n");
	}
    } else {
	print_and_exit (
	    "Error.  Input landmarks must be specified using either the "
	    "--input-xform option\nor the --fixed-landmarks and "
	    "--moving-landmarks option.\n");
    }

fprintf(fpdebug,"landmarks loaded ok  %s %s\n", 
		args_info->fixed_landmarks_arg, args_info->moving_landmarks_arg ); fflush(fpdebug);

fprintf(fpdebug,"trying to load input image %s\n", (const char*) args_info->input_image_arg); fflush(fpdebug);
FILE *ft;
ft = fopen(args_info->input_image_arg, "r");
if (!ft) {fprintf(fpdebug,"no such file\n"); fflush(fpdebug); } else fclose(ft);

    /* Load the input image */
    lw->m_input_img = plm_image_load_native (args_info->input_image_arg);
    if (!lw->m_input_img) {

		fprintf(fpdebug,"error loading input image %s\n", (const char*) args_info->input_image_arg); fflush(fpdebug);

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

fprintf(fpdebug,"enter do_landmark_warp\n"); fflush(fpdebug);

    lw = load_input_files (args_info);

fprintf(fpdebug,"input files loaded\n"); fflush(fpdebug);

    switch (args_info->algorithm_arg) {
    case algorithm_arg_tps:
	do_landmark_warp_itk_tps (lw);
	break;
    case algorithm_arg_gauss:
	do_landmark_warp_nsh (lw);
	break;
    case algorithm_arg_cone:
	do_landmark_warp_gcs (lw);
	break;
    default:
	break;
    }

fprintf(fpdebug,"warping complete\n"); fflush(fpdebug);

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
args_info_landmark_warp args_info;
	
//PARSE_ARGS comes from ...CLP.h
	PARSE_ARGS;

// write diagnostics just in case
// NSh: A fresh install of Windows 7 does not have c:\tmp
# if defined (_WIN32)
    char* fn = "C:/tmp/plastimatch-slicer-landwarp.txt";
# else
    char* fn = "/tmp/plastimatch-slicer-landwarp.txt";
# endif
    //FILE* 
		fpdebug = fopen (fn, "w");

	fprintf(fpdebug, "Parameters from Slicer GUI\n");
	fprintf(fpdebug, "RBF radius %f\n", plmslc_landwarp_rbf_radius);
	fprintf(fpdebug, "Stiffness %f\n", plmslc_landwarp_stiffness);
	fprintf(fpdebug, "Method %s\n", plmslc_landwarp_rbf_type.c_str());
	fflush(fpdebug);

	memset( &args_info, 0, sizeof(args_info));

// filling in args_info with data from Slicer
// plmslc_landwarp_nnnn come from .xml file via GenerateCLP and ..CLP.h

	args_info.input_image_arg = (char *)malloc(1024 * sizeof(char));
    strcpy(args_info.input_image_arg, plmslc_landwarp_moving_volume.c_str() );

	args_info.fixed_arg = (char *)malloc(1024 * sizeof(char));
	strcpy(args_info.fixed_arg, plmslc_landwarp_fixed_volume.c_str() );

	args_info.output_image_arg = (char *)malloc(1024 * sizeof(char));
    strcpy(args_info.output_image_arg, plmslc_landwarp_warped_volume.c_str() );

	args_info.radius_arg = plmslc_landwarp_rbf_radius;
	args_info.stiffness_arg = plmslc_landwarp_stiffness;

	args_info.algorithm_arg = algorithm_arg_cone; //default

	if (!strcmp(plmslc_landwarp_rbf_type.c_str(),"cone")) 
				args_info.algorithm_arg = algorithm_arg_cone;
	if (!strcmp(plmslc_landwarp_rbf_type.c_str(),"gauss")) 
				args_info.algorithm_arg = algorithm_arg_gauss;
	if (!strcmp(plmslc_landwarp_rbf_type.c_str(),"tps")) 
				args_info.algorithm_arg = algorithm_arg_tps;

// landmarks are passed from Slicer as lists, NOT filenames
// However, do_landmark_warp uses pointset_load(char *fn)
// To reuse code from landmark_warp, write landmarks 
// to temporary files for reading.

# if defined (_WIN32)
    char* fnfix = "C:/tmp/plmslc-landwarp-fixland.fscv";
# else
    char* fnfix = "/tmp/plmslc-landwarp-fixland.fcsv";
# endif
# if defined (_WIN32)
    char* fnmov = "C:/tmp/plmslc-landwarp-movland.fcsv";
# else
    char* fnmov = "/tmp/plmslc-landwarp-movland.fcsv";
# endif    

	// filling in args_info
	args_info.fixed_landmarks_arg = (char *)malloc(1024 * sizeof(char));
    strcpy(args_info.fixed_landmarks_arg, fnfix );
	args_info.moving_landmarks_arg = (char *)malloc(1024 * sizeof(char));
    strcpy(args_info.moving_landmarks_arg, fnmov );

	// writing landmarks
	FILE* fpfix = fopen (fnfix, "w");
	FILE* fpmov = fopen (fnmov, "w");

    unsigned long num_fiducials = plmslc_landwarp_fixed_fiducials.size();
    if (plmslc_landwarp_moving_fiducials.size() < num_fiducials) {
	num_fiducials = plmslc_landwarp_moving_fiducials.size();
    }

	/* NSh: pointset_load_fcsv assumes RAS, as does Slicer.
	For some reason, pointset_load_txt assumes LPS.
	Thus, we write out Slicer-style .fcsv
    */
	fprintf(fpfix, "# Fiducial List file FIX\n");
	fprintf(fpmov, "# Fiducial List file MOV\n");
	for (unsigned long i = 0; i < num_fiducials; i++) {
		fprintf(fpfix,"FIX%d,%f,%f,%f,1,1\n", i,
			plmslc_landwarp_fixed_fiducials[i][0],
			plmslc_landwarp_fixed_fiducials[i][1],
			plmslc_landwarp_fixed_fiducials[i][2] );
		fprintf(fpmov,"MOV%d,%f,%f,%f,1,1\n", i,
			plmslc_landwarp_moving_fiducials[i][0],
			plmslc_landwarp_moving_fiducials[i][1],
			plmslc_landwarp_moving_fiducials[i][2] );
	}
	fclose(fpfix);
	fclose(fpmov);

	fprintf(fpdebug, "wrote landmarks\n");
	fflush(fpdebug);

//	check_arguments (&args_info);
    do_landmark_warp (&args_info);

	fprintf(fpdebug,"done with do_landmark_warp\n");
	fflush(fpdebug);
	fclose(fpdebug);

	return EXIT_SUCCESS;
}
