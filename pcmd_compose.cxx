/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plm_config.h"
#include <time.h>
#include "itkImageRegionIterator.h"
#include "getopt.h"
#include "pcmd_compose.h"
#include "plm_image.h"

static void
compose_main (Compose_parms* parms)
{
    Plm_image *img;
    img = plm_image_load_native (parms->input_1);
    if (!img) {
	print_and_exit ("Error: could not open '%s' for read\n",
		       parms->input_1);
    }

    delete img;
}

static void
print_usage (void)
{
    printf (
	"Usage: plastimatch compose file_1 file_2 outfile\n"
	"\n"
	"Note:  file_1 is applied first, and then file_2.\n"
	"          outfile = file_2 o file_1\n"
	"          x -> x + file_2(x + file_1(x))\n"
    );
    exit (-1);
}

static void
compose_parse_args (Compose_parms* parms, int argc, char* argv[])
{
    int ch;
    static struct option longopts[] = {
	{ NULL,             0,                      NULL,           0 }
    };

    while ((ch = getopt_long(argc, argv, "", longopts, NULL)) != -1) {
	switch (ch) {
	default:
	    break;
	}
    }

    if (optind < argc) {
	strncpy (parms->input_1, argv[optind++], _MAX_PATH);
    }
    if (optind < argc) {
	strncpy (parms->input_2, argv[optind++], _MAX_PATH);
    }
    if (optind < argc) {
	strncpy (parms->output_fn, argv[optind++], _MAX_PATH);
    } else {
	print_usage ();
    }
}

void
do_command_compose (int argc, char *argv[])
{
    Compose_parms parms;
    
    compose_parse_args (&parms, argc, argv);

    compose_main (&parms);

    printf ("Finished!\n");
}
