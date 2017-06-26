/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plmdose_config.h"

#include "rt_spot_map.h"

class Rt_spot_map_private {
public:
    int i;
};

Rt_spot_map::Rt_spot_map ()
{
    d_ptr = new Rt_spot_map_private;
}

Rt_spot_map::~Rt_spot_map ()
{
    delete d_ptr;
}
