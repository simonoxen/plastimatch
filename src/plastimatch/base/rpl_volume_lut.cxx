/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plmbase_config.h"

#include "proj_volume.h"
#include "rpl_volume.h"
#include "rpl_volume_lut.h"
#include "volume.h"

class PLMBASE_API Rpl_volume_lut_private
{
public:
    Rpl_volume_lut_private ()
    {
    }
public:
    int i;
};

Rpl_volume_lut::Rpl_volume_lut ()
{
    d_ptr = new Rpl_volume_lut_private;
}

Rpl_volume_lut::~Rpl_volume_lut ()
{
    delete d_ptr;
}

void 
Rpl_volume_lut::build_lut (Rpl_volume *rv, Volume *vol)
{
    const Proj_volume* pv = rv->get_proj_volume ();

    /* Allocate memory - TBW */

    plm_long ijk[3];
    double xyz[3];
    LOOP_Z (ijk, xyz, vol) {
        LOOP_Y (ijk, xyz, vol) {
            LOOP_X (ijk, xyz, vol) {
                plm_long idx = vol->index (ijk);

                /* Initialize LUT to "no data" - TBW */

                /* Back project the voxel to the aperture plane */
                double ap_xy[2];
                pv->project (ap_xy, xyz);
                if (!is_number (ap_xy[0]) || !is_number (ap_xy[1])) {
                    continue;
                }

                /* Solve for interpolation fractions on aperture planes */
                plm_long ijk_f[3];
                //li_2d (ijk_f, li_frac_1, li_frac_2, ap_xy, ap_dim);
            }
        }
    }
}
