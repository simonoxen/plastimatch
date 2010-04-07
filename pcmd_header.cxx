/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plm_config.h"
#include <time.h>
#include "itkImageRegionIterator.h"
#include "getopt.h"
#include "pcmd_header.h"
#include "plm_image.h"
#include "plm_image_header.h"

static void
header_main (Header_parms* parms)
{
    Plm_image pli;
    Plm_image_header pih;

    pli.load_native (parms->mha_in_fn);
    pih.set_from_plm_image (&pli);
    pih.print ();
}

static void
header_print_usage (void)
{
    printf ("Usage: plastimatch header file\n"
	    );
    exit (-1);
}

static void
header_parse_args (Header_parms* parms, int argc, char* argv[])
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
	    header_print_usage ();
	}
    }
}

void
do_command_header (int argc, char *argv[])
{
    Header_parms parms;
    
    header_parse_args (&parms, argc, argv);

    header_main (&parms);
}
