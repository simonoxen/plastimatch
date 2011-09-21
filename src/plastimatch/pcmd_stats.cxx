/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plm_config.h"
#include <time.h>
#include "itkImageRegionIterator.h"
#include "itkVariableLengthVector.h"
#include "getopt.h"

#include "itk_image.h"
#include "itk_image_load.h"
#include "itk_image_stats.h"
#include "mha_io.h"
#include "pcmd_stats.h"
#include "plm_file_format.h"
#include "plm_image.h"
#include "plm_image_header.h"
#include "plm_int.h"
#include "proj_image.h"
#include "ss_img_stats.h"
#include "vf_stats.h"
#include "xform.h"

static void
stats_vf_main (Stats_parms* parms)
{
    Volume *vol = 0;
    Xform xf1, xf2;

    xform_load (&xf1, (const char*) parms->img_in_fn);

    if (xf1.m_type == XFORM_GPUIT_VECTOR_FIELD) {
	vol = xf1.get_gpuit_vf();
    }
    else if (xf1.m_type == XFORM_ITK_VECTOR_FIELD) {
	/* GCS FIX: This logic should be moved inside of xform class */
	int dim[3];
	float origin[3], spacing[3];
	Plm_image_header pih;

	pih.set_from_itk_image (xf1.get_itk_vf ());
	pih.get_origin (origin);
	pih.get_spacing (spacing);
	pih.get_dim (dim);

	xform_to_gpuit_vf (&xf2, &xf1, dim, origin, spacing);
	vol = xf2.get_gpuit_vf();
    }
    else 
    {
	print_and_exit ("Error: input file %s is not a vector field\n", 
	    (const char*) parms->img_in_fn);
    }

    if (vol->pix_type != PT_VF_FLOAT_INTERLEAVED) {
	fprintf (stderr, 
	    "Sorry, file \"%s\" is not an interleaved float vector field.\n", 
	    (const char*) parms->img_in_fn);
	fprintf (stderr, "Type = %d\n", vol->pix_type);
	delete vol;
	exit (-1);
    }

    if (parms->mask_fn.length() == 0) {
    	vf_analyze (vol);
    	vf_analyze_strain (vol);
	vf_analyze_jacobian (vol);
	vf_analyze_second_deriv (vol);
    }
    else {
	/* GCS FIX: Mask should be read as xform (to enable use of mhd) */
	Volume* mask = read_mha ((const char*) parms->mask_fn);
	vf_analyze (vol); 
	vf_analyze_strain (vol);
	vf_analyze_jacobian (vol);
	vf_analyze_second_deriv (vol);
	vf_analyze_mask (vol, mask);
	vf_analyze_strain_mask (vol, mask);
	delete mask;
    }
}

static void
stats_proj_image_main (Stats_parms* parms)
{
    Proj_image *proj;

    proj = proj_image_load ((const char*) parms->img_in_fn, 0);
    proj_image_debug_header (proj);
    proj_image_stats (proj);
    proj_image_destroy (proj);
}

static void
stats_ss_image_main (Stats_parms* parms)
{
    Plm_image plm ((const char*) parms->img_in_fn);

    if (plm.m_type != PLM_IMG_TYPE_ITK_UCHAR_VEC) {
	print_and_exit ("Failure loading file %s as ss_image.\n",
	    (const char*) parms->img_in_fn);
    }

    UCharVecImageType::Pointer img = plm.m_itk_uchar_vec;

    ss_img_stats (img);
}

static void
stats_img_main (Stats_parms* parms)
{
    FloatImageType::Pointer img = itk_image_load_float (
	(const char*) parms->img_in_fn, 0);

    double min_val, max_val, avg;
    int non_zero, num_vox;
    itk_image_stats (img, &min_val, &max_val, &avg, &non_zero, &num_vox);

    printf ("MIN %f AVE %f MAX %f NONZERO %d NUMVOX %d\n", 
	(float) min_val, (float) avg, (float) max_val, non_zero, num_vox);
}

static void
stats_main (Stats_parms* parms)
{
    switch (plm_file_format_deduce ((const char*) parms->img_in_fn)) {
    case PLM_FILE_FMT_VF:
	stats_vf_main (parms);
	break;
    case PLM_FILE_FMT_PROJ_IMG:
	stats_proj_image_main (parms);
	break;
    case PLM_FILE_FMT_SS_IMG_VEC:
	stats_ss_image_main (parms);
	break;
    case PLM_FILE_FMT_IMG:
    default:
	stats_img_main (parms);
	break;
    }
}

static void
stats_print_usage (void)
{
    printf ("Usage: plastimatch stats file [file ...]\n"
	    );
    exit (-1);
}

static void
stats_parse_args (Stats_parms* parms, int argc, char* argv[])
{
    int ch;
    static struct option longopts[] = {
	{ "input",          required_argument,      NULL,           2 },
	{ "mask",           required_argument,      NULL,           3 },
	{ NULL,             0,                      NULL,           0 }
    };

    while ((ch = getopt_long (argc, argv, "", longopts, NULL)) != -1) {
	switch (ch) {
	case 2:
	    parms->img_in_fn = optarg;
	    break;
	case 3:
	    parms->mask_fn = optarg;
	    break;
	default:
	    break;
	}
    }
    if (parms->img_in_fn.length() == 0) {
	optind ++;   /* Skip plastimatch command argument */
	if (optind < argc) {
	    parms->img_in_fn = argv[optind];
	} else {
	    printf ("Error: must specify input file\n");
	    stats_print_usage ();
	}
    }
    
    if (parms->mask_fn.length() == 0) {
	optind ++;
        if (optind < argc) 
	    parms->mask_fn = argv[optind];
    }
}

void
do_command_stats (int argc, char *argv[])
{
    Stats_parms parms;
    
    stats_parse_args (&parms, argc, argv);

    stats_main (&parms);
}
