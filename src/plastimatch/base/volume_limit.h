/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _volume_limit_h_
#define _volume_limit_h_

#include "plmbase_config.h"
#include "volume.h"

typedef struct volume_limit Volume_limit;
struct volume_limit {
    /* upper and lower limits of volume, including tolerances */
    double lower_limit[3];
    double upper_limit[3];

    /* dir == 0 if lower_limit corresponds to lower index */
    int dir[3];
};

enum point_location {
    POINTLOC_LEFT,
    POINTLOC_INSIDE,
    POINTLOC_RIGHT,
};
typedef enum point_location Point_location;

#endif
