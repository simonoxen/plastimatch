/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _pcmd_dvh_h_
#define _pcmd_dvh_h_

#include "plm_config.h"
#include "bstrwrap.h"
#include "resample_mha.h"

enum Dvh_units {
    DVH_UNITS_GY,
    DVH_UNITS_CGY,
};

enum Dvh_normalization {
    DVH_NORMALIZATION_PCT,
    DVH_NORMALIZATION_VOX,
};

class Dvh_parms {
public:
    CBString input_ss_img_fn;
    CBString input_ss_list_fn;
    CBString input_dose_fn;
    CBString output_csv_fn;
    enum Dvh_units input_units;
    enum Dvh_normalization normalization;
    int cumulative;
    int num_bins;
    float bin_width;
public:
    Dvh_parms () {
	input_units = DVH_UNITS_GY;
	normalization = DVH_NORMALIZATION_PCT;
	cumulative = 0;
	num_bins=256;
	bin_width=1;
    }
};

void do_command_dvh (int argc, char *argv[]);

#endif
