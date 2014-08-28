/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _pcmd_filter_h_
#define _pcmd_filter_h_

#include "plmcli_config.h"

class Filter_parms {
public:
    enum Filter_type {
        FILTER_TYPE_UNDEFINED,
        FILTER_TYPE_KERNEL,
        FILTER_TYPE_GAUSSIAN
    };

public:
    std::string in_image_fn;
    std::string out_image_fn;
    std::string ker_image_fn;

    float gauss_width;
    Filter_type filter_type;

public:
    Filter_parms () {
        filter_type = FILTER_TYPE_UNDEFINED;
        gauss_width = 10.f;
    }
};

void do_command_filter (int argc, char *argv[]);

#endif
