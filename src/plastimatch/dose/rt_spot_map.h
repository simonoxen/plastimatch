/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _rt_spot_map_h_
#define _rt_spot_map_h_

#include "plmdose_config.h"
#include "smart_pointer.h"

class Rt_spot_map_private;

class PLMDOSE_API Rt_spot_map {
public:
    SMART_POINTER_SUPPORT (Rt_spot_map);
    Rt_spot_map_private *d_ptr;
public:
    Rt_spot_map ();
    ~Rt_spot_map ();
};

#endif
