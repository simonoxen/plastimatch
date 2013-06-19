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
public:
    Gamma_parms () {
        dose_tolerance = 3.f;
        dta_tolerance = .03f;
    }
};

void do_command_gamma (int argc, char *argv[]);

#endif
