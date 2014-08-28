/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _volume_conv_h_
#define _volume_conv_h_

#include "plmbase_config.h"

PLMBASE_API Volume::Pointer 
volume_conv (
    const Volume::Pointer& vol_in,
    const Volume::Pointer& ker_in
);

#endif
