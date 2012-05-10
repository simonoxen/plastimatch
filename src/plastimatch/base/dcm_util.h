/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _dcm_util_h_
#define _dcm_util_h_

#include "plmbase_config.h"
#include <string>

class Slice_index;

std::string dcm_anon_patient_id (void);

void dcm_load_rdd (Slice_index* rdd, const char* dicom_dir);

void dcm_get_date_time (
    std::string *date,
    std::string *time
);
PLMBASE_C_API char*
dcm_uid (char *uid, const char *uid_root);

#endif
