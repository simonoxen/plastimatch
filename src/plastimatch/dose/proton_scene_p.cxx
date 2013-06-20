/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plm_config.h"
#include "proton_scene_p.h"

Proton_scene_private::Proton_scene_private ()
{
    debug = false;
    step_length = 0.;
    patient = Plm_image::New();
    target = Plm_image::New();
    ap = Aperture::New();
}

Proton_scene_private::~Proton_scene_private ()
{
}
