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

    /* Required input */
    strcpy (parms.input_fn, input_dicomrt_ss.c_str());

    /* Optional inputs */
    if (output_labelmap.compare ("None") != 0) {
	strcpy (parms.output_labelmap_fn, output_labelmap.c_str());
    }
    if (output_dose.compare ("None") != 0) {
	strcpy (parms.output_dose_img, output_dose.c_str());
    }
    if (output_image.compare ("None") != 0) {
	strcpy (parms.output_img, output_image.c_str());
    }
    if (reference_vol.compare ("None") != 0) {
	strcpy (parms.fixed_im_fn, reference_vol.c_str());
    }

    /* Process warp */
    file_type = PLM_FILE_FMT_DICOM_DIR;
    rtds_warp (&rtds, file_type, &parms);

    return EXIT_SUCCESS;
}
