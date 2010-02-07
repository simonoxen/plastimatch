#include <stdio.h>
#include <iostream>
#include <vector>
#include "dicomrt-import-slicerCLP.h"

#include "plm_config.h"
#include "plm_file_format.h"
#include "rtds.h"
#include "rtds_warp.h"
#include "warp_parms.h"

int 
main (int argc, char * argv [])
{
    PARSE_ARGS;

    Plm_file_format file_type;
    Warp_parms parms;
    Rtds rtds;

    strcpy (parms.input_fn, input_dicomrt_ss.c_str());
    strcpy (parms.output_labelmap_fn, output_labelmap.c_str());
    strcpy (parms.fixed_im_fn, reference_vol.c_str());

    /* Process warp */
    file_type = PLM_FILE_FMT_DICOM_RTSS;
    rtds_warp (&rtds, file_type, &parms);

    return EXIT_SUCCESS;
}
