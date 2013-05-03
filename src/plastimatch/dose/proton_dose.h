/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _proton_dose_h_
#define _proton_dose_h_

#include "plmdose_config.h"
#include "proton_scene.h"

class Proton_scene;
class Volume;

PLMDOSE_C_API
Volume*
proton_dose_compute (Proton_scene::Pointer& scene);

#endif
