/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include <time.h>
#include "plm_config.h"
#include "itkImage.h"
#include "itkLinearInterpolateImageFunction.h"

#include "getopt.h"
#include "xf_to_xf_main.h"
#include "itk_image.h"
#include "print_and_exit.h"
#include "xform.h"

void
xf_to_xf_main (Xf_To_Xf_Parms* parms)
{
    Xform xf_in, xf_out;

    load_xform (&xf_in, parms->xf_in_fn);
    xform_to_itk_vf (&xf_out, &xf_in, parms->dim, parms->offset, parms->spacing);
    save_xform (&xf_out, parms->vf_out_fn);
}

void
print_usage (void)
{
    printf ("Usage: xf_to_xf --input=xform_in --output=vf_out --dims=\"x y z\"\n");
    printf ("          --offset=\"x y z\" --spacing=\"x y z\"\n");
    exit (-1);
}

void
parse_args (Xf_To_Xf_Parms* parms, int argc, char* argv[])
{
    int ch, rc;
    static struct option longopts[] = {
	{ "input",          required_argument,      NULL,           1 },
	{ "output",         required_argument,      NULL,           2 },
	{ "dims",           required_argument,      NULL,           3 },
	{ "offset",         required_argument,      NULL,           4 },
	{ "spacing",        required_argument,      NULL,           5 },
	{ NULL,             0,                      NULL,           0 }
    };

    while ((ch = getopt_long (argc, argv, "", longopts, NULL)) != -1) {
	switch (ch) {
	case 1:
	    strncpy (parms->xf_in_fn, optarg, _MAX_PATH);
	    break;
	case 2:
	    strncpy (parms->vf_out_fn, optarg, _MAX_PATH);
	    break;
	case 3:
	    rc = sscanf (optarg, "%d %d %d", &(parms->dim[0]), 
		    &(parms->dim[1]), &(parms->dim[2]));
	    if (rc != 3) {
		print_usage();
	    }
	    break;
	case 4:
	    rc = sscanf (optarg, "%g %g %g", &(parms->offset[0]), 
		    &(parms->offset[1]), &(parms->offset[2]));
	    if (rc != 3) {
		print_usage();
	    }
	    break;
	case 5:
	    rc = sscanf (optarg, "%g %g %g", &(parms->spacing[0]), 
		    &(parms->spacing[1]), &(parms->spacing[2]));
	    if (rc != 3) {
		print_usage();
	    }
	    break;
	default:
	    break;
	}
    }
    if (!parms->xf_in_fn[0] || !parms->vf_out_fn[0] || !parms->dim[0] || parms->spacing[0] == 0.0 || parms->offset[0] == 0.0) {
		printf ("Error: must specify all options\n");
		print_usage();
    }
}

int
main(int argc, char *argv[])
{
    Xf_To_Xf_Parms parms;
    
    parse_args (&parms, argc, argv);

    xf_to_xf_main (&parms);

    printf ("Finished!\n");
    return 0;
}
