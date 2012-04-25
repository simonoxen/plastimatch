/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _rpl_volume_h_
#define _rpl_volume_h_

#include "plm_config.h"
#include "proj_matrix.h"
#include "volume.h"

typedef struct rpl_volume Rpl_volume;
struct rpl_volume {
    Volume *vol;
    Proj_matrix *pmat;
    double *depth_offset;
    double cam[3];
    double ap_ul_room[3];
    double incr_r[3];
    double incr_c[3];
    double ray_step;
};

#endif
