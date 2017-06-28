/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plmdose_config.h"
#include <list>

#include "rpl_volume.h"
#include "rt_dij.h"
#include "volume_macros.h"

void 
Rt_dij::set_from_rpl_dose (
    const plm_long ij[2],
    size_t energy_index,
    const Rpl_volume *rpl_dose, 
    Volume::Pointer& dose_vol)
{
    this->rows.push_back (Rt_dij_row (
            float (ij[0]), float (ij[1]), float (energy_index)));
    Rt_dij_row& rt_dij_row = this->rows.back ();

    const Volume *dose = dose_vol.get();
    plm_long ijk[3];
    double xyz[3];
    LOOP_Z (ijk, xyz, dose) {
        LOOP_Y (ijk, xyz, dose) {
            LOOP_X (ijk, xyz, dose) {
                plm_long idx = dose->index (ijk);
                float val = rpl_dose->get_rgdepth (xyz);
                rt_dij_row.dose.push_back (Rt_dij_dose (idx, val));
            }
        }
    }
}
