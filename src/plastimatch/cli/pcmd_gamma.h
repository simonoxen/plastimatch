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
    std::string ref_image_fn;
    std::string cmp_image_fn;
    std::string out_image_fn;

    /* Gamma options */
    float dose_tolerance;
    float dta_tolerance;
    bool have_reference_dose;
    float reference_dose;
    float gamma_max;
public:
    Gamma_parms () {
        dose_tolerance = .03f;
        dta_tolerance = 3.f;
        have_reference_dose = false;
        reference_dose = 0.f;
        gamma_max = 2.0f;
    }
};

void do_command_gamma (int argc, char *argv[]);

#endif
