/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plm_config.h"
#include <stdlib.h>
#include "itkImageRegionIterator.h"
#include "getopt.h"
#include "pcmd_segment.h"
#include "pcmd_segment_body_ggo.h"
#include "plm_image.h"
#include "plm_image_header.h"
#include "plm_ggo.h"
#include "resample_mha.h"


static void
print_usage ()
{
    printf (
	"Usage: plastimatch segment algorithm [options]\n"
	"Algorithms:\n"
	"  body\n"
	""
	"Type \"plastimatch segment algorithm --help\" to see additional options\n"
    );

    exit (0);
}

static void
do_segment_body (args_info_pcmd_segment_body *args_info)
{
}

void
do_command_segment_body (int argc, char *argv[])
{
    GGO (pcmd_segment_body, args_info, 2);

    do_segment_body (&args_info);

    GGO_FREE (pcmd_segment_body, args_info, 2);
}

void
do_command_segment (int argc, char *argv[])
{
    char* command;

    if (argc == 2) {
	print_usage ();
    }

    command = argv[2];
    if (!strcmp (command, "body")) {
	do_command_segment_body (argc, argv);
    }
    else {
	print_usage ();
    }
}
