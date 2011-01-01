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
#include "segment_body.h"

static void
print_usage ()
{
    printf (
	"Usage: plastimatch segment algorithm [options]\n"
	"Algorithms:\n"
	"  body\n"
	""
	"Type \"plastimatch segment algorithm --help\" for additional options\n"
    );
    exit (0);
}

static void
do_segment_body (args_info_pcmd_segment_body *args_info)
{
    Segment_body sb;

    /* Load the input image */
    sb.img_in.load_native (args_info->input_arg);

    /* Set other parameter(s) */
    sb.bot_given = args_info->bot_given;
    sb.bot = args_info->bot_arg;

    /* Do segmentation */
    sb.do_segmentation ();

    /* Save output file */
    sb.img_out.save_image (args_info->output_arg);
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
