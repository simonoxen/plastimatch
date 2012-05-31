/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plm_config.h"
#include <iostream>
#include <vector>
#include <stdio.h>
#include "plastimatch_slicer_dvhCLP.h"

#include "plmutil.h"

int 
main (int argc, char * argv [])
{
    PARSE_ARGS;

    Dvh_parms_pcmd dvh;

    /* Required input */
    dvh.input_ss_img_fn = input_ss_image.c_str();
    dvh.input_dose_fn = input_dose_image.c_str();
    dvh.output_csv_fn = output_dvh_filename.c_str();

    if (histogram_type == "Cumulative") {
        dvh.dvh_parms.cumulative = 1;
    } else {
        dvh.dvh_parms.cumulative = 0;
    }
    dvh.dvh_parms.num_bins = num_bins;
    dvh.dvh_parms.bin_width = bin_width;

    /* Optional inputs */
    if (input_ss_list != "" && input_ss_list != "None") {
        dvh.input_ss_list_fn = input_ss_list.c_str();
    }

    /* Process DVH */
    dvh_execute (&dvh);

    return EXIT_SUCCESS;
}
