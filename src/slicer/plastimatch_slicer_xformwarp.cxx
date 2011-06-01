/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plm_config.h"
#include <algorithm>
#include <stdio.h>
#include <iostream>
#include <vector>
#include <time.h>

#include "plastimatch_slicer_xformwarpCLP.h"

#include "plm_file_format.h"
#include "rtds.h"
#include "rtds_warp.h"
#include "xform.h"
#include "warp_parms.h"

int 
main (int argc, char * argv [])
{
    PARSE_ARGS;

    Warp_parms parms;
    Plm_file_format file_type;
    Rtds rtds;

    /* Input image (required) */
    parms.input_fn = plmslc_xformwarp_input_img.c_str();

    /* Get xform either from MRML scene or file */
    if (plmslc_xformwarp_input_xform_s != "" 
	&& plmslc_xformwarp_input_xform_s != "NONE")
    {
	parms.xf_in_fn = plmslc_xformwarp_input_xform_s.c_str();
    }
    else if (plmslc_xformwarp_input_xform_f != "" 
	&& plmslc_xformwarp_input_xform_f != "NONE")
    {
	parms.xf_in_fn = plmslc_xformwarp_input_xform_f.c_str();
    }

    /* Output image (required) */
    parms.output_img_fn = plmslc_xformwarp_output_img.c_str();

    /* What is the input file type? */
    file_type = plm_file_format_deduce ((const char*) parms.input_fn);

    /* Process warp */
    rtds_warp (&rtds, file_type, &parms);

    return EXIT_SUCCESS;
}
