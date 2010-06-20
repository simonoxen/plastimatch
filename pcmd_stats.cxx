/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plm_config.h"
#include <time.h>
#include "itkImageRegionIterator.h"
#include "getopt.h"
#include "itk_image.h"
#include "mha_io.h"
#include "plm_file_format.h"
#include "proj_image.h"
#include "pcmd_stats.h"
#include "vf_stats.h"

static void
stats_vf_main (Stats_parms* parms)
{
    Volume* vol;

    vol = read_mha (parms->mha_in_fn);

    if (!vol) {
	fprintf (stderr, "Sorry, couldn't open file \"%s\" for read.\n", 
	    parms->mha_in_fn);
	exit (-1);
    }

    if (vol->pix_type != PT_VF_FLOAT_INTERLEAVED) {
	fprintf (stderr, "Sorry, file \"%s\" is not an interleaved "
	    "float vector field.\n", parms->mha_in_fn);
	fprintf (stderr, "Type = %d\n", vol->pix_type);
	volume_destroy (vol);
	exit (-1);
    }

    if (parms->mask_fn[0] == '\0') {
    	vf_analyze (vol);
    	vf_analyze_strain (vol);
    	volume_destroy (vol);
    }
    else {
	Volume* mask = read_mha (parms->mask_fn);
	vf_analyze (vol); 
	vf_analyze_strain (vol);
	vf_analyze_mask (vol, mask);
	vf_analyze_strain_mask (vol, mask);
	volume_destroy (vol);
	volume_destroy (mask);
    }
}

static void
stats_proj_image_main (Stats_parms* parms)
{
    Proj_image *proj;

    proj = proj_image_load (parms->mha_in_fn, 0);
    proj_image_debug_header (proj);
    proj_image_stats (proj);
    proj_image_destroy (proj);
}

static void
stats_img_main (Stats_parms* parms)
{

    typedef itk::ImageRegionIterator< FloatImageType > FloatIteratorType;
    FloatImageType::Pointer img = itk_image_load_float (parms->mha_in_fn, 0);
    FloatImageType::RegionType rg = img->GetLargestPossibleRegion ();
    FloatIteratorType it (img, rg);

    int first = 1;
    float min_val, max_val;
    int num = 0;
    double sum = 0.0;

    for (it.GoToBegin(); !it.IsAtEnd(); ++it) {
	float v = it.Get();
	if (first) {
	    min_val = max_val = v;
	    first = 0;
	}
	if (min_val > v) min_val = v;
	if (max_val < v) max_val = v;
	sum += v;
	num ++;
    }

    printf ("MIN %f AVE %f MAX %f NUM %d\n",
	    min_val, (float) (sum / num), max_val, num);
}

static void
stats_main (Stats_parms* parms)
{
    switch (plm_file_format_deduce (parms->mha_in_fn)) {
    case PLM_FILE_FMT_VF:
	stats_vf_main (parms);
	break;
    case PLM_FILE_FMT_PROJ_IMG:
	stats_proj_image_main (parms);
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
	{ NULL,             0,                      NULL,           0 }
    };

    while ((ch = getopt_long (argc, argv, "", longopts, NULL)) != -1) {
	switch (ch) {
	case 2:
	    strncpy (parms->mha_in_fn, optarg, _MAX_PATH);
	    parms->mask_fn[0] = '\0';
	    break;
	case 3:
	    strncpy (parms->mha_in_fn, optarg, _MAX_PATH);
	    strncpy (parms->mask_fn, optarg, _MAX_PATH);
	    break;
	default:
	    break;
	}
    }
    if (!parms->mha_in_fn[0]) {
	optind ++;   /* Skip plastimatch command argument */
	if (optind < argc) {
	    strncpy (parms->mha_in_fn, argv[optind], _MAX_PATH);
	} else {
	    printf ("Error: must specify input file\n");
	    stats_print_usage ();
	}
    }
    
    if (!parms->mask_fn[0]) {
	optind ++;
        if (optind < argc) 
	   strncpy(parms->mask_fn, argv[optind], _MAX_PATH);
    }
}

void
do_command_stats (int argc, char *argv[])
{
    Stats_parms parms;
    
    stats_parse_args (&parms, argc, argv);

    stats_main (&parms);
}
