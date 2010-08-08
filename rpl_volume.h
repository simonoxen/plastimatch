/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _rpl_volume_h_
#define _rpl_volume_h_

#include "plm_config.h"
#include "volume.h"

typedef struct rpl_volume Rpl_volume;
struct rpl_volume {
    Volume *vol;
    double *depth_offset;
    double p1[3];
    double ap_ul_room[3];
    double incr_r[3];
    double incr_c[3];
    double ray_step;
};

#if defined __cplusplus
extern "C" {
#endif

gpuit_EXPORT 
Rpl_volume*
rpl_volume_create (
    Volume* ct_vol,       // ct volume
    int ires[2],          // aperture dimensions
    double p1[3],         // position of source
    double ap_ul_room[3], // position of aperture in room coords
    double incr_r[3],     // change in room coordinates for each ap pixel
    double incr_c[3],     // change in room coordinates for each ap pixel
    float ray_step        // uniform ray step size
);
gpuit_EXPORT 
void
rpl_volume_destroy (Rpl_volume *rpl_vol);
gpuit_EXPORT 
void
rpl_volume_save (Rpl_volume *rpl_vol, char *filename);
gpuit_EXPORT 
void
rpl_volume_compute (
    Rpl_volume *rpl_vol,   /* I/O: this gets filled in with depth info */
    Volume *ct_vol         /* I:   the ct volume */
);
gpuit_EXPORT 
double
rpl_volume_get_rgdepth (
    Rpl_volume *rpl_vol,   /* I: volume of radiological depths */
    double* ct_xyz,        /* I: location of voxel in world space */
    Proj_matrix *pmat
);

#if defined __cplusplus
}
#endif

#endif
