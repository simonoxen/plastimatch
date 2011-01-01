/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plm_config.h"
#include <time.h>
#include "itkImageRegionIterator.h"
#include "getopt.h"
#include "pcmd_segment.h"
#include "pcmd_segment_body_ggo.h"
#include "plm_image.h"
#include "plm_image_header.h"
#include "plm_ggo.h"
#include "resample_mha.h"

static void
do_segment_body (args_info_pcmd_segment *args_info)
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
    do_command_segment_body (argc, argv);
}
