/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _translation_score_h_
#define _translation_score_h_

#include "plmregister_config.h"
#include "volume.h"

float
translation_score (
    const Volume::Pointer& fixed,
    const Volume::Pointer& moving,
    const float dxyz[3]);

#endif
