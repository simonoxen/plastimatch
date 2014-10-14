/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plm_config.h"
#include "RTP_plan.h"
#include "RTP_plan_p.h"

RTP_plan_private::RTP_plan_private ()
{
    debug = false;
    step_length = 0.;
    smearing = 0.f;
    patient = Plm_image::New();
    target = Plm_image::New();
    ap = Aperture::New();
	normalization_dose = 1.0;
}

RTP_plan_private::~RTP_plan_private ()
{
}
