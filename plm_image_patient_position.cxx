/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plm_config.h"
#include <string.h>
#include "plm_image_patient_position.h"
#include <stdio.h>
#include <stdlib.h>
Plm_image_patient_position
plm_image_patient_position_parse (const char* string)
{
    if (!strcmp (string,"HFS ")) {
	return PATIENT_POSITION_HFS;
    }
    else if (!strcmp (string,"HFP ")) {
	return PATIENT_POSITION_HFP;
    } else {
	return PATIENT_POSITION_UNKNOWN;
    }
}
