/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plm_config.h"
#include <time.h>
#include "itkImageRegionIterator.h"
#include "getopt.h"
#include "adjust_main.h"
#include "plm_image.h"

static void
adjust_main (Adjust_Parms* parms)
{
    typedef itk::ImageRegionIterator< FloatImageType > FloatIteratorType;

    Plm_image *plm_image = plm_image_load (parms->mha_in_fn, 
					  PLM_IMG_TYPE_ITK_FLOAT);
    FloatImageType::Pointer img = plm_image->m_itk_float;
    FloatImageType::RegionType rg = img->GetLargestPossibleRegion ();
    FloatIteratorType it (img, rg);

    if (parms->have_truncate_above) {
	for (it.GoToBegin(); !it.IsAtEnd(); ++it) {
	    float v = it.Get();
	    if (v > parms->truncate_above) {
		it.Set (parms->truncate_above);
	    }
	}
    }

    if (parms->have_truncate_below) {
	for (it.GoToBegin(); !it.IsAtEnd(); ++it) {
	    float v = it.Get();
	    if (v < parms->truncate_below) {
		it.Set (parms->truncate_below);
	    }
	}
    }

    if (parms->have_ab_scale) {
	it.GoToBegin();
	for (it.GoToBegin(); !it.IsAtEnd(); ++it) {
	    float v = it.Get();
	    float d_per_fx = v / parms->num_fx;
	    v = v * (parms->alpha_beta + d_per_fx) 
		/ (parms->alpha_beta + parms->norm_dose_per_fx);
	    it.Set (v);
	}
    }

    if (parms->have_scale) {
	it.GoToBegin();
	for (it.GoToBegin(); !it.IsAtEnd(); ++it) {
	    float v = it.Get();
	    v = v * parms->scale;
	    it.Set (v);
	}
    }

    if (parms->have_stretch) {
	float vmin, vmax;
	it.GoToBegin();
	vmin = it.Get();
	vmax = it.Get();
	for (it.GoToBegin(); !it.IsAtEnd(); ++it) {
	    float v = it.Get();
	    if (v > vmax) {
		vmax = v;
	    } else if (v < vmin) {
		vmin = v;
	    }
	}
	for (it.GoToBegin(); !it.IsAtEnd(); ++it) {
	    float v = it.Get();
	    v = (v - vmin) / (vmax - vmin);
	    v = (v + parms->stretch[0]) 
		* (parms->stretch[1] - parms->stretch[0]);
	    it.Set (v);
	}
    }

    if (parms->output_dicom) {
	itk_image_save_short_dicom (img, parms->mha_out_fn);
    } else {
	if (parms->output_type) {
	    plm_image->convert (parms->output_type);
	}
	plm_image->save_image (parms->mha_out_fn);
    }

    delete plm_image;
}

static void
adjust_print_usage (void)
{
    printf ("Usage: plastimatch adjust [options]\n"
	    "Required:\n"
	    "    --input=image_in\n"
	    "    --output=image_out\n"
	    "Optional:\n"
	    "    --output-type={uchar,short,ushort,ulong,float}\n"
	    "    --scale=\"min max\"\n"
	    "    --ab-scale=\"ab nfx ndf\"       (Alpha-beta scaling)\n"
	    "    --stretch=\"min max\"\n"
	    "    --truncate-above=value\n"
	    "    --truncate-below=value\n"
	    );
    exit (-1);
}

static void
adjust_parse_args (Adjust_Parms* parms, int argc, char* argv[])
{
    int ch;
    static struct option longopts[] = {
	{ "input",          required_argument,      NULL,           2 },
	{ "output",         required_argument,      NULL,           3 },
	{ "truncate_above", required_argument,      NULL,           4 },
	{ "truncate-above", required_argument,      NULL,           4 },
	{ "truncate_below", required_argument,      NULL,           5 },
	{ "truncate-below", required_argument,      NULL,           5 },
	{ "stretch",        required_argument,      NULL,           6 },
	{ "output-format",  required_argument,      NULL,           7 },
	{ "output_format",  required_argument,      NULL,           7 },
	{ "output-type",    required_argument,      NULL,           8 },
	{ "output_type",    required_argument,      NULL,           8 },
	{ "scale",          required_argument,      NULL,           9 },
	{ "ab_scale",       required_argument,      NULL,           10 },
	{ "ab-scale",       required_argument,      NULL,           10 },
	{ NULL,             0,                      NULL,           0 }
    };

    while ((ch = getopt_long(argc, argv, "", longopts, NULL)) != -1) {
	int rc;
	switch (ch) {
	case 2:
	    strncpy (parms->mha_in_fn, optarg, _MAX_PATH);
	    break;
	case 3:
	    strncpy (parms->mha_out_fn, optarg, _MAX_PATH);
	    break;
	case 4:
	    if (sscanf (optarg, "%f", &parms->truncate_above) != 1) {
		printf ("Error: truncate_above takes an argument\n");
		adjust_print_usage ();
	    }
	    parms->have_truncate_above = 1;
	    break;
	case 5:
	    if (sscanf (optarg, "%f", &parms->truncate_below) != 1) {
		printf ("Error: truncate_below takes an argument\n");
		adjust_print_usage ();
	    }
	    parms->have_truncate_below = 1;
	    break;
	case 6:
	    if (sscanf (optarg, "%f %f", &parms->stretch[0], &parms->stretch[1]) != 2) {
		printf ("Error: stretch takes two arguments\n");
		adjust_print_usage ();
	    }
	    parms->have_stretch = 1;
	    break;
	case 7:
	    if (!strcmp (optarg, "dicom")) {
		parms->output_dicom = 1;
	    } else {
		fprintf (stderr, "Error.  --output-format option only supports dicom.\n");
		adjust_print_usage ();
	    }
	    break;
	case 8:
	    parms->output_type = plm_image_type_parse (optarg);
	    if (parms->output_type == PLM_IMG_TYPE_UNDEFINED) {
		adjust_print_usage();
	    }
	    break;
	case 9:
	    if (sscanf (optarg, "%f", &parms->scale) != 1) {
		printf ("Error: --scale takes an arguments\n");
		adjust_print_usage ();
	    }
	    parms->have_scale = 1;
	    break;
	case 10:
	    rc = sscanf (optarg, "%f %f %f", 
		&parms->alpha_beta, 
		&parms->num_fx, 
		&parms->norm_dose_per_fx);
	    if (rc != 3) {
		printf ("Error: --ab-scale takes 3 arguments\n");
		adjust_print_usage ();
	    }
	    parms->have_ab_scale = 1;
	    break;
	default:
	    break;
	}
    }
    if (!parms->mha_in_fn[0] || !parms->mha_out_fn[0]) {
	printf ("Error: must specify --input and --output\n");
	adjust_print_usage ();
    }
}

void
do_command_adjust (int argc, char *argv[])
{
    Adjust_Parms parms;
    
    adjust_parse_args (&parms, argc, argv);

    adjust_main (&parms);

    printf ("Finished!\n");
}
