/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _pcmd_adjust_h_
#define _pcmd_adjust_h_

#include "plmcli_config.h"
#include <string>
#include <string.h>
#include <stdlib.h>
#include "plm_image_type.h"

class Adjust_parms {
public:
    std::string img_in_fn;
    std::string img_out_fn;
    std::string img_ref_fn;

    /* Piecewise linear adjustment */
    std::string pw_linear;

    /* Alpha-beta scaling */
    float alpha_beta;
    float num_fx;
    float norm_dose_per_fx;
    bool have_ab_scale;

    /* histogram matching */
    bool do_hist_match;
    bool hist_th;
    int hist_levels;
    int hist_points;



    bool output_dicom;
    Plm_image_type output_type;
public:
    Adjust_parms () {
	have_ab_scale = false;
	do_hist_match = false;
	output_dicom = false;
	output_type = PLM_IMG_TYPE_UNDEFINED;
    }
};

void do_command_adjust (int argc, char *argv[]);

#endif
