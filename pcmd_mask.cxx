/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plm_config.h"
#include <time.h>
#include "itkImageRegionIterator.h"
#include "getopt.h"
#include "pcmd_mask.h"
#include "mask_mha.h"
#include "plm_image.h"

static void
mask_main (Mask_Parms* parms)
{
    Plm_image *img;
    img = plm_image_load_native ((const char*) parms->input_fn);
    if (!img) {
	print_and_exit ("Error: could not open '%s' for read\n",
	    (const char*) parms->input_fn);
    }

    UCharImageType::Pointer mask = itk_image_load_uchar (parms->mask_fn, 0);

    switch (img->m_type) {
    case PLM_IMG_TYPE_ITK_UCHAR:
	img->m_itk_uchar = mask_image (img->m_itk_uchar, mask, 
	    parms->negate_mask, parms->mask_value);
	break;
    case PLM_IMG_TYPE_ITK_SHORT:
	img->m_itk_short = mask_image (img->m_itk_short, mask, 
	    parms->negate_mask, parms->mask_value);
	break;
    case PLM_IMG_TYPE_ITK_USHORT:
	img->m_itk_ushort = mask_image (img->m_itk_ushort, mask, 
	    parms->negate_mask, parms->mask_value);
	break;
    case PLM_IMG_TYPE_ITK_ULONG:
	img->m_itk_uint32 = mask_image (img->m_itk_uint32, mask, 
	    parms->negate_mask, parms->mask_value);
	break;
    case PLM_IMG_TYPE_ITK_FLOAT:
	img->m_itk_float = mask_image (img->m_itk_float, mask, 
	    parms->negate_mask, parms->mask_value);
	break;
    default:
	print_and_exit ("Unhandled conversion in mask_main\n");
	break;
    }

    if (parms->output_dicom) {
	img->save_short_dicom ((const char*) parms->output_fn);
    } else {
	if (parms->output_type) {
	    img->convert (parms->output_type);
	}
	img->save_image ((const char*) parms->output_fn);
    }

    delete img;
}

static void
mask_print_usage (void)
{
    printf ("Usage: plastimatch mask [options]\n"
	    "Required:\n"
	    "    --input=image_in\n"
	    "    --output=image_out\n"
	    "    --mask=mask_image_in\n"
	    "Optional:\n"
	    "    --negate-mask\n"
	    "    --mask-value=float\n"
	    "    --output-format=dicom\n"
	    "    --output-type={uchar,short,ushort,ulong,float}\n"
	    );
    exit (-1);
}

static void
mask_parse_args (Mask_Parms* parms, int argc, char* argv[])
{
    int ch;
    static struct option longopts[] = {
	{ "input",          required_argument,      NULL,           2 },
	{ "output",         required_argument,      NULL,           3 },
	{ "mask",           required_argument,      NULL,           4 },
	{ "negate-mask",    no_argument,            NULL,           5 },
	{ "negate_mask",    no_argument,            NULL,           5 },
	{ "mask-value",     required_argument,      NULL,           6 },
	{ "mask_value",     required_argument,      NULL,           6 },
	{ "output-format",  required_argument,      NULL,           7 },
	{ "output_format",  required_argument,      NULL,           7 },
	{ "output-type",    required_argument,      NULL,           8 },
	{ "output_type",    required_argument,      NULL,           8 },
	{ NULL,             0,                      NULL,           0 }
    };

    while ((ch = getopt_long(argc, argv, "", longopts, NULL)) != -1) {
	switch (ch) {
	case 2:
	    parms->input_fn = optarg;
	    break;
	case 3:
	    parms->output_fn = optarg;
	    break;
	case 4:
	    parms->mask_fn = optarg;
	    break;
	case 5:
	    parms->negate_mask = true;
	    break;
	case 6:
	    if (sscanf (optarg, "%f", &parms->mask_value) != 1) {
		printf ("Error: mask_value takes an argument\n");
		mask_print_usage ();
	    }
	    break;
	case 7:
	    if (!strcmp (optarg, "dicom")) {
		parms->output_dicom = true;
	    } else {
		fprintf (stderr, 
		    "Error.  --output-format option only supports dicom.\n");
		mask_print_usage ();
	    }
	    break;
	case 8:
	    parms->output_type = plm_image_type_parse (optarg);
	    if (parms->output_type == PLM_IMG_TYPE_UNDEFINED) {
		mask_print_usage();
	    }
	    break;
	default:
	    break;
	}
    }
    if (parms->input_fn.length() == 0 
	|| parms->output_fn.length() == 0 
	|| parms->mask_fn.length() == 0)
    {
	printf ("Error: must specify --input, --output, and --mask\n");
	mask_print_usage ();
    }
}

void
do_command_mask (int argc, char *argv[])
{
    Mask_Parms parms;
    
    mask_parse_args (&parms, argc, argv);

    mask_main (&parms);

    printf ("Finished!\n");
}
