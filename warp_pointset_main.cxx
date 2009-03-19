/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plm_config.h"
#include "warp_pointset_main.h"
#include "getopt.h"
#include "itk_pointset.h"
#include "xform.h"

void
warp_pointset_main (Warp_Pointset_Parms* parms)
{
    Xform xf;
    PointSetType::Pointer ps_in = PointSetType::New ();

    pointset_load (ps_in, parms->ps_in_fn);
    pointset_debug (ps_in);

    load_xform (&xf, parms->xf_in_fn);

    PointSetType::Pointer ps_out = pointset_warp (ps_in, &xf);
    pointset_debug (ps_out);
}

void
print_usage (void)
{
    printf ("Usage: warp_pointset --input=ps_in  --xf=xf_in --output=ps_out\n");
    exit (-1);
}

void
parse_args (Warp_Pointset_Parms* parms, int argc, char* argv[])
{
    int ch;
    int have_offset = 0;
    int have_spacing = 0;
    int have_dims = 0;
    static struct option longopts[] = {
	{ "input",          required_argument,      NULL,           2 },
	{ "output",         required_argument,      NULL,           3 },
	{ "xf",             required_argument,      NULL,           4 },
	{ NULL,             0,                      NULL,           0 }
    };

    while ((ch = getopt_long(argc, argv, "", longopts, NULL)) != -1) {
	switch (ch) {
	case 2:
	    strncpy (parms->ps_in_fn, optarg, _MAX_PATH);
	    break;
	case 3:
	    strncpy (parms->ps_out_fn, optarg, _MAX_PATH);
	    break;
	case 4:
	    strncpy (parms->xf_in_fn, optarg, _MAX_PATH);
	    break;
	default:
	    break;
	}
    }
    if (!parms->ps_in_fn[0] || !parms->ps_out_fn[0] || !parms->xf_in_fn[0]) {
	printf ("Error: must specify --input, --output, and --vf or --xf\n");
	print_usage();
    }
}

int
main(int argc, char *argv[])
{
    Warp_Pointset_Parms parms;
    
    parse_args (&parms, argc, argv);

    warp_pointset_main (&parms);

    printf ("Finished!\n");
    return 0;
}
