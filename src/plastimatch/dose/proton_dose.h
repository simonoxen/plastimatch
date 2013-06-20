/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _proton_dose_h_
#define _proton_dose_h_

#include "plmdose_config.h"
#include "plm_image.h"
#include "proton_scene.h"
#include "proton_scene_p.h"

PLMDOSE_API
Plm_image::Pointer
proton_dose_compute (Proton_scene::Pointer& scene);

#endif
