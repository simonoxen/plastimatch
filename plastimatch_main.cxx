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
#include "add_main.h"
#include "adjust_main.h"
#include "compare_main.h"
#include "crop_main.h"
#include "diff_main.h"
#include "dvh_main.h"
#include "mask_main.h"
#include "pcmd_header.h"
#include "pcmd_segment.h"
#include "plm_registration.h"
#include "plm_stages.h"
#include "plm_version.h"
#include "resample_main.h"
#include "stats_main.h"
#include "warp_main.h"


/* GCS FIX: "segment" is a hidden option until it works */
static void
print_usage (int return_code)
{
    printf ("plastimatch version %s\n", PLASTIMATCH_VERSION_STRING);
    printf (
	"Usage: plastimatch command [options]\n"
	"Commands:\n"
	"  add         "
	"  adjust      "
	"  crop        "
	"  compare     "
	"\n"
	"  convert     "
	"  diff        "
	"  dvh         "
	"  header      "
	"\n"
	"  mask        "
	"  register    "
	"  resample    "
	"  stats       "
	"\n"
	"  warp        "
	"\n"
	"\n"
	"For detailed usage of a specific command, type:\n"
	"  plastimatch command\n"
    );
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
	    exit (1);
	}
    } else {
	command_filename = argv[1];
    }

    if (plm_parms_parse_command_file (&regp, command_filename) < 0) {
	print_usage (1);
    }
    do_registration (&regp);
}

void
do_command (int argc, char* argv[])
{
    char* command;

    if (argc == 1) {
	print_usage (0);
    }
    command = argv[1];

    if (!strcmp (command, "add")) {
	do_command_add (argc, argv);
    }
    else if (!strcmp (command, "adjust")) {
	do_command_adjust (argc, argv);
    }
    else if (!strcmp (command, "compare")) {
	do_command_compare (argc, argv);
    }
    else if (!strcmp (command, "convert")) {
	/* warp and convert are the same */
	do_command_warp (argc, argv);
    }
    else if (!strcmp (command, "crop")) {
	do_command_crop (argc, argv);
    }
    else if (!strcmp (command, "diff")) {
	do_command_diff (argc, argv);
    }
    else if (!strcmp (command, "dvh")) {
	do_command_dvh (argc, argv);
    }
    else if (!strcmp (command, "header")) {
	do_command_header (argc, argv);
    }
    else if (!strcmp (command, "mask")) {
	do_command_mask (argc, argv);
    }
    else if (!strcmp (command, "register")) {
	do_command_register (argc, argv);
    }
    else if (!strcmp (command, "resample")) {
	do_command_resample (argc, argv);
    }
    else if (!strcmp (command, "segment")) {
	do_command_segment (argc, argv);
    }
    else if (!strcmp (command, "stats")) {
	do_command_stats (argc, argv);
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
main (int argc, char *argv[])
{
    do_command (argc, argv);

    return 0;
}
