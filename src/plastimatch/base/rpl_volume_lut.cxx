/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plmbase_config.h"

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
}
