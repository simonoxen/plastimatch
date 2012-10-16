/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plm_config.h"

#include "proj_volume.h"

class Proj_volume_private {
public:
    Volume *vol;
    Proj_matrix *pmat;
    double *depth_offset;
    double cam[3];
    double ap_ul_room[3];
    double incr_r[3];
    double incr_c[3];
    double ray_step;
};

Proj_volume::Proj_volume () {
    d_ptr = new Proj_volume_private;
}

Proj_volume::~Proj_volume () {
    delete d_ptr;
}

