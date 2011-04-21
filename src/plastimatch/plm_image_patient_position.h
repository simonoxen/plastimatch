/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _plm_image_patient_position_h_
#define _plm_image_patient_position_h_

#include "plm_config.h"

enum Plm_image_patient_position {
    PATIENT_POSITION_UNKNOWN,
    PATIENT_POSITION_HFS,
    PATIENT_POSITION_HFP,
    PATIENT_POSITION_FFS,
    PATIENT_POSITION_FFP,
};

plastimatch1_EXPORT
Plm_image_patient_position
plm_image_patient_position_parse (const char* string);

#endif
