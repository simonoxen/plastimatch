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

#if defined __cplusplus
extern "C" {
#endif

gpuit_EXPORT
void
volume_limit_set (Volume_limit *vol_limit, Volume *vol);

#if defined __cplusplus
}
#endif

#endif
