/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plm_config.h"
#include "photon_plan.h"
#include "photon_plan_p.h"

Photon_plan_private::Photon_plan_private ()
{
    debug = false;
    step_length = 0.;
    smearing = 0.f;
    patient = Plm_image::New();
    target = Plm_image::New();
    ap = Aperture::New();
	source_size = 0.;
}

Photon_plan_private::~Photon_plan_private ()
{
}
