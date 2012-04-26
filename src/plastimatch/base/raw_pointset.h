/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _raw_pointset_h_
#define _raw_pointset_h_

#include "plmbase_config.h"

typedef struct raw_pointset Raw_pointset;
struct raw_pointset {
    int num_points;
    float *points;
};

#endif
