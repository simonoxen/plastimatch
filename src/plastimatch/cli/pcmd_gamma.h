/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _pcmd_gamma_h_
#define _pcmd_gamma_h_

#include "plmcli_config.h"
#include <string.h>
#include <stdlib.h>
#include "pstring.h"
#include "plm_image_type.h"

class Gamma_parms {
public:
    std::string ref_image_fn; //should be compatible with mha, dcm, OPG(IBA text)
    std::string cmp_image_fn; //should be compatible with mha, dcm, OPG(IBA text)
    std::string out_image_fn; //gamma map: mha will be enough	

    /* Gamma options */
    float dose_tolerance;
    float dta_tolerance;
    bool have_reference_dose;
    float reference_dose;
    float gamma_max;

	/* Extended Gamma options by YK*/	
	std::string out_report_fn; //YK: text file name
	bool b_local_gamma; // if true, local dose difference will be used for perc. dose difference
	bool b_skip_low_dose_gamma; // if true, gamma will not be calculated for points below cut-off dose e.g. <10%
	float f_inherent_resample_mm; //if -1.0, no resample will be carried out
	float f_analysis_threshold_perc; //if -1.0, no threshold will be applied

public:
    Gamma_parms () {
        dose_tolerance = .03f;
        dta_tolerance = 3.f;
        have_reference_dose = false;
        reference_dose = 0.f;
        gamma_max = 2.0f;

		//out_report_fn = "";
		b_local_gamma = false; // if true, local dose difference will be used for perc. dose difference
		b_skip_low_dose_gamma = true; // if true, gamma will not be calculated for points below cut-off dose e.g. <10%
		f_inherent_resample_mm = -1.0; //if -1.0, no resample will be carried out
		f_analysis_threshold_perc = -1.0; //if -1.0, no threshold will be applied
    }
};

void do_command_gamma (int argc, char *argv[]);

#endif
