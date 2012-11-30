/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _proton_dose_h_
#define _proton_dose_h_

#include "plmdose_config.h"
#include "threading.h"
#include "plm_path.h"

class Proton_Scene;
class Volume;

PLMDOSE_C_API
Volume*
proton_dose_compute (Proton_Scene* scene);

#endif
