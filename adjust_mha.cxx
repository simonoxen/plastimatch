/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plm_config.h"
#include <time.h>
#include "itkImageRegionIterator.h"
#include "getopt.h"
#include "adjust_mha.h"
#include "itk_image.h"

void
adjust_mha_main (Adjust_Mha_Parms* parms)
{
    typedef itk::ImageRegionIterator< FloatImageType > FloatIteratorType;

    FloatImageType::Pointer img = load_float (parms->mha_in_fn);
    FloatImageType::RegionType rg = img->GetLargestPossibleRegion ();
    FloatIteratorType it (img, rg);

    if (parms->have_upper_trunc) {
	for (it.GoToBegin(); !it.IsAtEnd(); ++it) {
	    float v = it.Get();
	    if (v > parms->upper_trunc) {
		it.Set (parms->upper_trunc);
	    }
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
	    v = (v + parms->stretch[0]) * (parms->stretch[1] - parms->stretch[0]);
	    it.Set (v);
	}
    }

    save_float (img, parms->mha_out_fn);
}

void
print_usage (void)
{
    printf ("Usage: adjust_mha --input=image_in --output=image_out [options]\n");
    printf ("Opts:    --upper_trunc=value\n");
    printf ("Opts:    --stretch=\"min max\"\n");
    exit (-1);
}

void
parse_args (Adjust_Mha_Parms* parms, int argc, char* argv[])
{
    int ch;
    int have_offset = 0;
    int have_spacing = 0;
    int have_dims = 0;
    static struct option longopts[] = {
	{ "input",          required_argument,      NULL,           2 },
	{ "output",         required_argument,      NULL,           3 },
	{ "upper_trunc",    required_argument,      NULL,           4 },
	{ "stretch",        required_argument,      NULL,           5 },
	{ NULL,             0,                      NULL,           0 }
    };

    while ((ch = getopt_long(argc, argv, "", longopts, NULL)) != -1) {
	switch (ch) {
	case 2:
	    strncpy (parms->mha_in_fn, optarg, _MAX_PATH);
	    break;
	case 3:
	    strncpy (parms->mha_out_fn, optarg, _MAX_PATH);
	    break;
	case 4:
	    if (sscanf (optarg, "%f", &parms->upper_trunc) != 1) {
		printf ("Error: upper_trunc takes an argument\n");
		print_usage();
	    }
	    parms->have_upper_trunc = 1;
	    break;
	case 5:
	    if (sscanf (optarg, "%f %f", &parms->stretch[0], &parms->stretch[1]) != 1) {
		printf ("Error: stretch takes two arguments\n");
		print_usage();
	    }
	    parms->have_stretch = 1;
	    break;
	default:
	    break;
	}
    }
    if (!parms->mha_in_fn[0] || !parms->mha_out_fn[0]) {
	printf ("Error: must specify --input and --output\n");
	print_usage();
    }
}

int
main(int argc, char *argv[])
{
    Adjust_Mha_Parms parms;
    
    parse_args (&parms, argc, argv);

    adjust_mha_main (&parms);

    printf ("Finished!\n");
    return 0;
}
