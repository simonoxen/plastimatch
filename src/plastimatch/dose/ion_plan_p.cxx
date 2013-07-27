/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plm_config.h"
#include "ion_plan.h"
#include "ion_plan_p.h"

Ion_plan_private::Ion_plan_private ()
{
    debug = false;
    step_length = 0.;
    smearing = 0.f;
    patient = Plm_image::New();
    target = Plm_image::New();
    ap = Aperture::New();
}

Ion_plan_private::~Ion_plan_private ()
{
}
