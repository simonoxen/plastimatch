/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _volume_fill_h_
#define _volume_fill_h_

#include "plmbase_config.h"
#include "volume.h"

PLMBASE_API 
template<class T> 
void volume_fill (
    Volume* vol,
    T val
);

#endif
