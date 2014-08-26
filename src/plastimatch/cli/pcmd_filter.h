/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _pcmd_filter_h_
#define _pcmd_filter_h_

#include "plmcli_config.h"

class Filter_parms {
public:
    std::string in_image_fn;
    std::string out_image_fn;
    std::string ker_image_fn;

    float gauss_width;
public:
    Filter_parms () {
        gauss_width = 10.f;
    }
};

void do_command_filter (int argc, char *argv[]);

#endif
