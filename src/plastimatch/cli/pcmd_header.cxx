/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plmcli_config.h"
#include <time.h>
#include "itkImageRegionIterator.h"

#include "plmbase.h"

#include "getopt.h"
#include "pcmd_header.h"

static void
header_main (Header_parms* parms)
{
    Plm_image pli;

    pli.load_native ((const char*) parms->img_in_fn);
    pli.print ();
}

static void
header_print_usage (void)
{
    printf ("Usage: plastimatch header input-file\n"
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
	    parms->img_in_fn = optarg;
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
