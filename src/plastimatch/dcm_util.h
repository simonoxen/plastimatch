/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _dcm_util_h_
#define _dcm_util_h_

#include "plm_config.h"
#include <string>

#if DCMTK_FOUND && DCMTK_VERSION_36 && PLM_CONFIG_PREFER_DCMTK
#  define PLM_DCM_USE_DCMTK 1
#else
#  undef PLM_DCM_USE_DCMTK
#endif

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
