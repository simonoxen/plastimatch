/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plm_config.h"
#include <algorithm>
#include <string>
#include <string.h>
#include "plm_image_patient_position.h"
#include <stdio.h>
#include <stdlib.h>
Plm_image_patient_position
plm_image_patient_position_parse (const char* string)
{
    /* Convert to upper case and trim spaces */
    std::string patient_pos;
    patient_pos = string;
    std::transform(patient_pos.begin(), patient_pos.end(), patient_pos.begin(), toupper);
    patient_pos.erase(0 , patient_pos.find_first_not_of(" ") );
    patient_pos.erase(patient_pos.find_last_not_of(" ") + 1);

    if (patient_pos == "HFS") {
	return PATIENT_POSITION_HFS;
    } else if (patient_pos == "HFP") {
	return PATIENT_POSITION_HFP;
    } else if (patient_pos == "FFS") {
	return PATIENT_POSITION_FFS;
    } else if (patient_pos == "FFP") {
	return PATIENT_POSITION_FFP;
    } else {
	return PATIENT_POSITION_UNKNOWN;
    }
}
