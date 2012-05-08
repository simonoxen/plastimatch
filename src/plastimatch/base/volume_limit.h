/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _volume_limit_h_
#define _volume_limit_h_

/**
*  You probably do not want to #include this header directly.
 *
 *   Instead, it is preferred to #include "plmbase.h"
 */

#include "plmbase_config.h"

enum point_location {
    POINTLOC_LEFT,
    POINTLOC_INSIDE,
    POINTLOC_RIGHT,
};
typedef enum point_location Point_location;

class Volume_limit {
public:
    /* upper and lower limits of volume, including tolerances */
    double lower_limit[3];
    double upper_limit[3];

    /* dir == 0 if lower_limit corresponds to lower index */
    int dir[3];
};

C_API int volume_limit_clip_ray (
        Volume_limit *vol_limit,    /* INPUT:  The bounding box to clip to */
        double *ip1,                /* OUTPUT: Intersection point 1 */
        double *ip2,                /* OUTPUT: Intersection point 2 */
        double *p1,                 /* INPUT:  Starting point of ray */
        double *ray                 /* INPUT:  Direction of ray */
);
C_API int volume_limit_clip_segment (
        Volume_limit *vol_limit,    /* INPUT:  The bounding box to clip to */
        double *ip1,                /* OUTPUT: Intersection point 1 */
        double *ip2,                /* OUTPUT: Intersection point 2 */
        double *p1,                 /* INPUT:  Line segment point 1 */
        double *p2                  /* INPUT:  Line segment point 2 */
);
C_API void volume_limit_set (Volume_limit *vol_limit, Volume *vol);


#endif
