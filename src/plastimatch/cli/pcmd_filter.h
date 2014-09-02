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
        FILTER_TYPE_GABOR,
        FILTER_TYPE_GAUSSIAN,
        FILTER_TYPE_KERNEL
    };

public:
    std::string in_image_fn;
    std::string in_kernel_fn;
    std::string out_image_fn;
    std::string out_kernel_fn;

    Filter_type filter_type;
    float gauss_width;
    int gabor_uv[2];

public:
    Filter_parms () {
        filter_type = FILTER_TYPE_UNDEFINED;
        gauss_width = 10.f;
        gabor_uv[0] = 0;
        gabor_uv[1] = 0;
    }
};

void do_command_filter (int argc, char *argv[]);

#endif
