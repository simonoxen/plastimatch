/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plm_config.h"
#include <time.h>
#include <stdlib.h>
#include <string.h>
#if defined (HAVE_GETOPT_LONG)
#include <getopt.h>
#else
#include "getopt.h"
#endif
#include "rad_registration.h"
#include "itk_image.h"
#include "itk_optim.h"
#include "xform.h"
#include "version.h"

#define BUFLEN 2048


static void
print_usage (void)
{
    printf ("plastimatch version %s\n", RAD_VERSION_STRING);
    printf ("Usage: plastimatch options_file\n");
    exit (-1);
}

void
parse_args (Registration_Parms* regp, int argc, char* argv[])
{
    if (argc != 2) {
	print_usage ();
    }
    if (parse_command_file (regp, argv[1]) < 0) {
	print_usage ();
    }
}

int
main(int argc, char *argv[])
{
    Registration_Parms regp;

    parse_args (&regp, argc, argv);
    do_registration (&regp);
    printf ("Finished!\n");

    return 0;
}
