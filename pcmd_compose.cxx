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
    img = plm_image_load_native (parms->xf_in_1_fn);
    if (!img) {
	print_and_exit ("Error: could not open '%s' for read\n",
	    (const char*) parms->xf_in_1_fn);
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
    if (argc != 5) {
	print_usage ();
    }
    
    parms->xf_in_1_fn = argv[2];
    parms->xf_in_2_fn = argv[3];
    parms->xf_out_fn = argv[4];
}

void
do_command_compose (int argc, char *argv[])
{
    Compose_parms parms;
    
    compose_parse_args (&parms, argc, argv);

    compose_main (&parms);

    printf ("Finished!\n");
}
