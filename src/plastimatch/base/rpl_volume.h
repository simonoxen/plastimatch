/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _rpl_volume_h_
#define _rpl_volume_h_

/**
*  You probably do not want to #include this header directly.
 *
 *   Instead, it is preferred to #include "plmbase.h"
 */

#include "plmbase_config.h"

class Proj_matrix;
class Volume;

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

C_API void rpl_volume_compute (
        Rpl_volume *rpl_vol,   /* I/O: this gets filled in with depth info */
        Volume *ct_vol         /* I:   the ct volume */
);
C_API Rpl_volume* rpl_volume_create (
        Volume* ct_vol,       // ct volume
        Proj_matrix *pmat,    // projection matrix from source to aperture
        int ires[2],          // aperture dimensions
        double cam[3],        // position of source
        double ap_ul_room[3], // position of aperture in room coords
        double incr_r[3],     // change in room coordinates for each ap pixel
        double incr_c[3],     // change in room coordinates for each ap pixel
        float ray_step        // uniform ray step size
);
C_API void rpl_volume_destroy (Rpl_volume *rpl_vol);
C_API double rpl_volume_get_rgdepth (
        Rpl_volume *rpl_vol,   /* I: volume of radiological depths */
        double* ct_xyz         /* I: location of voxel in world space */
);
C_API void rpl_volume_save (Rpl_volume *rpl_vol, char *filename);


#endif
