/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _rpl_volume_lut_h_
#define _rpl_volume_lut_h_

#include "plmbase_config.h"
#include "smart_pointer.h"

class Rpl_volume;
class Rpl_volume_lut_private;
class Volume;

class PLMBASE_API Rpl_volume_lut 
{
public:
    SMART_POINTER_SUPPORT (Rpl_volume_lut);
    Rpl_volume_lut_private *d_ptr;
public:
    Rpl_volume_lut ();
    ~Rpl_volume_lut ();
public:
    void build_lut (Rpl_volume *rv, Volume *vol);
};

#endif
