/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plmcli_config.h"
#include <stdlib.h>
#include <string.h>

#include "file_util.h"
#include "pcmd_add.h"
#include "pcmd_adjust.h"
#include "pcmd_autolabel.h"
#include "pcmd_autolabel_train.h"
#include "pcmd_compare.h"
#include "pcmd_compose.h"
#include "pcmd_crop.h"
#include "pcmd_dice.h"
#include "pcmd_diff.h"
#include "pcmd_drr.h"
#include "pcmd_dvh.h"
#include "pcmd_xio_dvh.h"
#include "pcmd_mabs.h"
#include "pcmd_mask.h"
#include "pcmd_header.h"
#include "pcmd_jacobian.h"
#include "pcmd_probe.h"
#include "pcmd_resample.h"
#include "pcmd_scale.h"
#include "pcmd_script.h"
#include "pcmd_segment.h"
#include "pcmd_stats.h"
#include "pcmd_synth.h"
#include "pcmd_synth_vf.h"
#include "pcmd_thumbnail.h"
#include "pcmd_warp.h"
#include "pcmd_xf_convert.h"
#include "plm_parms.h"
#include "plm_stages.h"
#include "plm_version.h"
#include "print_and_exit.h"

static void
print_version (void)
{
    printf ("plastimatch version %s\n", PLASTIMATCH_VERSION_STRING);
}

static void
print_usage (int return_code)
{
    printf ("plastimatch version %s\n", PLASTIMATCH_VERSION_STRING);
    printf (
        "Usage: plastimatch command [options]\n"
        "Commands:\n"
        "  add         "
        "  adjust      "
        "  average     "
//        "  autolabel   "
        "  crop        "
        "  compare     "
        "\n"
        "  compose     "
        "  convert     "
        "  dice        "
        "  diff        "
//        "  drr         "
        "  dvh         "
        "\n"
        "  fill        "
        "  header      "
	"  jacobian    "
//        "  mask        "
        "  mask        "
        "  probe       "
        "\n"
        "  register    "
        "  resample    "
        "  scale       "
        "  segment     "
        "  stats       "
        "\n"
        "  synth       "
        "  synth-vf    "
        "  thumbnail   "
        "  warp        "
        "  xf-convert  "
//        "  xio-dvh     "
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
    char *command_filename;
    Registration_parms *regp = new Registration_parms;

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

    if (plm_parms_parse_command_file (regp, command_filename) < 0) {
        print_usage (1);
    }
    do_registration (regp);
    delete regp;
}

void
do_command (int argc, char* argv[])
{
    char* command;

    if (argc == 1) {
        print_usage (0);
    }
    command = argv[1];

    if (!strcmp (command, "--version")) {
        print_version ();
    }
    else if (!strcmp (command, "add")) {
        /* add and average are the same */
        do_command_add (argc, argv);
    }
    else if (!strcmp (command, "adjust")) {
        do_command_adjust (argc, argv);
    }
    else if (!strcmp (command, "average")) {
        /* add and average are the same */
        do_command_add (argc, argv);
    }
    else if (!strcmp (command, "autolabel")) {
        do_command_autolabel (argc, argv);
    }
    else if (!strcmp (command, "autolabel-train")) {
        do_command_autolabel_train (argc, argv);
    }
    else if (!strcmp (command, "compare")) {
        do_command_compare (argc, argv);
    }
    else if (!strcmp (command, "compose")) {
        do_command_compose (argc, argv);
    }
    else if (!strcmp (command, "convert")) {
        /* convert and warp are the same */
        do_command_warp (argc, argv);
    }
    else if (!strcmp (command, "crop")) {
        do_command_crop (argc, argv);
    }
    else if (!strcmp (command, "dice")) {
        do_command_dice (argc, argv);
    }
    else if (!strcmp (command, "diff")) {
        do_command_diff (argc, argv);
    }
    else if (!strcmp (command, "drr")) {
        do_command_drr (argc, argv);
    }
    else if (!strcmp (command, "dvh")) {
        do_command_dvh (argc, argv);
    }
    else if (!strcmp (command, "header")) {
        do_command_header (argc, argv);
    }
    else if (!strcmp (command, "jacobian")) {
        do_command_jacobian (argc, argv);
    }
    else if (!strcmp (command, "fill")) {
        /* fill and mask are the same */
        do_command_mask (argc, argv);
    }
    else if (!strcmp (command, "mabs")) {
        do_command_mabs (argc, argv);
    }
    else if (!strcmp (command, "mask")) {
        /* fill and mask are the same */
        do_command_mask (argc, argv);
    }
    else if (!strcmp (command, "probe")) {
        do_command_probe (argc, argv);
    }
    else if (!strcmp (command, "register")) {
        do_command_register (argc, argv);
    }
    else if (!strcmp (command, "resample")) {
        do_command_resample (argc, argv);
    }
    else if (!strcmp (command, "scale")) {
        do_command_scale (argc, argv);
    }
    else if (!strcmp (command, "script")) {
        do_command_script (argc, argv);
    }
    else if (!strcmp (command, "segment")) {
        do_command_segment (argc, argv);
    }
    else if (!strcmp (command, "slice")) {
        print_and_exit ("Error: slice command is now called thumbnail.\n");
    }
    else if (!strcmp (command, "stats")) {
        do_command_stats (argc, argv);
    }
    else if (!strcmp (command, "synth")) {
        do_command_synth (argc, argv);
    }
    else if (!strcmp (command, "synth-vf")) {
        do_command_synth_vf (argc, argv);
    }
    else if (!strcmp (command, "thumbnail")) {
        do_command_thumbnail (argc, argv);
    }
    else if (!strcmp (command, "warp")) {
        /* convert and warp are the same */
        do_command_warp (argc, argv);
    }
    else if (!strcmp (command, "xf-convert")) {
        do_command_xf_convert (argc, argv);
    }
    else if (!strcmp (command, "xio-dvh")) {
        do_command_xio_dvh (argc, argv);
    }
    else if (argc == 2) {
        if (!file_exists (argv[1])) {
            print_usage (1);
        }
        /* Older usage, just "plastimatch parms.txt" */
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
