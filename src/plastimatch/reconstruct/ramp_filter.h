/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _ramp_filter_h_
#define _ramp_filter_h_

#include "plm_config.h"

#if defined __cplusplus
extern "C" {
#endif

void
ramp_filter (
    float *data, 
    unsigned int width, 
    unsigned int height
);

#if defined __cplusplus
}
#endif

#endif
