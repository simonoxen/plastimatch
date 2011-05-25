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

    /* Parse command line parameters */
    //plm_clp_parse (&parms, &parse_fn, &usage_fn, argc, argv, 1);

    parms.input_fn = plmslc_xformwarp_input_img.c_str();
    parms.xf_in_fn = plmslc_xformwarp_input_xform.c_str();
    parms.output_img_fn = plmslc_xformwarp_output_img.c_str();

    /*NSh: removed Dij processing */

    /* What is the input file type? */
    file_type = plm_file_format_deduce ((const char*) parms.input_fn);

    /* Pointsets are a special case 
    if (file_type == PLM_FILE_FMT_POINTSET) {
	warp_pointset_main (&parms);
	return EXIT_SUCCESS;
    } */

    /* Process warp */
    rtds_warp (&rtds, file_type, &parms);

    return EXIT_SUCCESS;
}
