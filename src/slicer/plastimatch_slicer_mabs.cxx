/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include <stdio.h>
#include <iostream>
#include <vector>
#include "plastimatch_slicer_mabsCLP.h"

#include "mabs.h"
#include "mabs_parms.h"

int 
main (int argc, char * argv [])
{
    PARSE_ARGS;

    Mabs_parms parms;

    /* Required input */
    strncpy (parms.atlas_dir, atlas_directory.c_str(), _MAX_PATH-1);
    strncpy (parms.labeling_input_fn, 
        input_image.c_str(), _MAX_PATH-1);

    /* GCS FIX: Need registration config */

    /* Output image */
    strncpy (parms.labeling_output_fn, output_image.c_str(), _MAX_PATH-1);

    /* Process mabs */
    Mabs mabs;
    mabs.run (parms);

    return EXIT_SUCCESS;
}
