/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plm_config.h"
#include <time.h>
#include "itkImageRegionIterator.h"
#include "getopt.h"
#include "pcmd_segment.h"
#include "plm_image.h"
#include "plm_image_header.h"
#include "resample_mha.h"

static void
segment_main (Segment_parms* parms)
{
    Plm_image *pli;
    Plm_image_header pih;
    float origin[3];
    float center[3];
    float spacing[3];
    int dim[3];
    int d;

    /* Load image */
    pli = plm_image_load (
	(const char*) parms->img_in_fn, PLM_IMG_TYPE_ITK_FLOAT);

    /* Compute dimensions of thumbnail */
    pih.set_from_plm_image (pli);
    pih.print ();

    pih.get_image_center (center);
    for (d = 0; d < 2; d++) {
	origin[d] = center[d] 
	    - parms->thumbnail_spacing * (parms->thumbnail_dim - 1) / 2;
	spacing[d] = parms->thumbnail_spacing;
	dim[d] = parms->thumbnail_dim;
    }
    origin[2] = center[2];
    spacing[2] = spacing[0];
    dim[2] = 1;
    
    /* Resample the image */
    pli->m_itk_float = resample_image (
	pli->m_itk_float, origin, spacing, dim, -1000, 1);

    /* Save the output file */
    pli->save_image ((const char*) parms->img_out_fn);

    delete (pli);
}

static void
segment_print_usage (void)
{
    printf (
	"Usage: plastimatch [options] segment input-file\n"
	"Options:\n"
	"  --input file\n"
	"  --thumbnail-dim size\n"
	"  --thumbnail-spacing size\n"
	    );
    exit (-1);
}

static void
segment_parse_args (Segment_parms* parms, int argc, char* argv[])
{
    int rc, ch;
    static struct option longopts[] = {
	{ "input",          required_argument,      NULL,           1 },
	{ "thumbnail-dim",  required_argument,      NULL,           2 },
	{ "thumbnail_dim",  required_argument,      NULL,           2 },
	{ "thumbnail-spacing",required_argument,    NULL,           3 },
	{ "thumbnail_spacing",required_argument,    NULL,           3 },
	{ "output",         required_argument,      NULL,           4 },
	{ NULL,             0,                      NULL,           0 }
    };

    while ((ch = getopt_long (argc, argv, "", longopts, NULL)) != -1) {
	switch (ch) {
	case 1:
	    parms->img_in_fn = optarg;
	    break;
	case 2:
	    rc = sscanf (optarg, "%d", &parms->thumbnail_dim);
	    if (rc != 1) {
		printf ("Error: %s requires an argument", argv[optind]);
		segment_print_usage ();
	    }
	    break;
	case 3:
	    rc = sscanf (optarg, "%f", &parms->thumbnail_spacing);
	    if (rc != 1) {
		printf ("Error: %s requires an argument", argv[optind]);
		segment_print_usage ();
	    }
	    break;
	case 4:
	    parms->img_out_fn = optarg;
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
	    segment_print_usage ();
	}
    }
}

void
do_command_segment (int argc, char *argv[])
{
    Segment_parms parms;
    
    segment_parse_args (&parms, argc, argv);

    segment_main (&parms);
}
