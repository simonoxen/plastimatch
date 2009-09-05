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
#include "plm_registration.h"
#include "itk_image.h"
#include "itk_optim.h"
#include "xform.h"
#include "plm_version.h"
#include "warp_main.h"

static void
print_usage (int return_code)
{
    printf ("plastimatch version %s\n", PLASTIMATCH_VERSION_STRING);
    printf ("Usage: plastimatch command [options]\n");
    printf ("Commands:\n");
    printf ("  register\n");
    printf ("  warp\n");
    exit (return_code);
}

void
do_command_register (int argc, char* argv[])
{
    char * command_filename;
    Registration_Parms regp;

    if (!strcmp (argv[1], "register")) {
	if (argc > 2) {
	    command_filename = argv[2];
	} else {
	    printf ("Usage: plastimatch register command_file\n");
	    print_usage (1);
	}
    } else {
	command_filename = argv[1];
    }

    if (parse_command_file (&regp, argv[1]) < 0) {
	print_usage (1);
    }
    do_registration (&regp);
}

void
do_command (int argc, char* argv[])
{
    char* command = argv[1];

    if (argc == 1) {
	print_usage (0);
    }

    if (!strcmp (command, "register")) {
	do_command_register (argc, argv);
    }
    else if (!strcmp (command, "warp")) {
	do_command_warp (argc, argv);
    }
    else if (argc == 2) {
	/* Older usage */
	do_command_register (argc, argv);
    }
    else {
	print_usage (1);
    }
}

int
main(int argc, char *argv[])
{
    do_command (argc, argv);

    return 0;
}
