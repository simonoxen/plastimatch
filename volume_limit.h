/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _volume_limit_h_
#define _volume_limit_h_

#include "plm_config.h"
#include "volume.h"

typedef struct volume_limit Volume_limit;
struct volume_limit {
    /* upper and lower limits of volume, including tolerances */
    double limits[3][2];

    /* dir == 0 if limits go from low to high */
    int dir[3];
};

typedef enum point_location Point_location;
enum point_location {
    POINTLOC_LEFT,
    POINTLOC_INSIDE,
    POINTLOC_RIGHT,
};

#if defined __cplusplus
extern "C" {
#endif

gpuit_EXPORT
void
volume_limit_set (Volume_limit *vol_limit, Volume *vol);
gpuit_EXPORT
int
volume_limit_clip_segment (
    Volume_limit *vol_limit,    /* INPUT:  The bounding box to clip to */
    double *ip1,                /* OUTPUT: Intersection point 1 */
    double *ip2,                /* OUTPUT: Intersection point 2 */
    double *p1,                 /* INPUT:  Line segment point 1 */
    double *p2                  /* INPUT:  Line segment point 2 */
);
gpuit_EXPORT
int
volume_limit_clip_ray (
    Volume_limit *vol_limit,    /* INPUT:  The bounding box to clip to */
    double *ip1,                /* OUTPUT: Intersection point 1 */
    double *ip2,                /* OUTPUT: Intersection point 2 */
    double *p1,                 /* INPUT:  Starting point of ray */
    double *ray                 /* INPUT:  Direction of ray */
);

#if defined __cplusplus
}
#endif

#endif
