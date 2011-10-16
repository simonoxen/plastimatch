/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plm_config.h"

#include "pointset.h"
#include "registration_data.h"

Registration_data::Registration_data ()
{
    fixed_image = 0;
    moving_image = 0;
    fixed_landmarks = 0;
    moving_landmarks = 0;
}

Registration_data::~Registration_data ()
{
    if (fixed_landmarks) delete fixed_landmarks;
    if (moving_landmarks) delete moving_landmarks;
}
