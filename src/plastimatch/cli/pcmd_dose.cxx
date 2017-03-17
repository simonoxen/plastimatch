/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plmcli_config.h"

#include "logfile.h"
#include "pcmd_dose.h"
#include "plm_return_code.h"
#include "registration.h"

void
do_command_dose (int argc, char* argv[])
{

    if (argc < 2) {
        printf ("Usage: plastimatch dose command_file\n");
        exit (1);
    }

    char *command_filename = argv[2];

    printf ("Not fully implemented.  Sorry.\n");
}
