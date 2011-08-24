/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _dcm_util_h_
#define _dcm_util_h_

#include "plm_config.h"
#include <string>

class Referenced_dicom_dir;

std::string 
dcm_anon_patient_id (void);
void
dcm_load_rdd (Referenced_dicom_dir* rdd, const char* dicom_dir);
void
dcm_get_date_time (
    std::string *date,
    std::string *time
);

#endif
