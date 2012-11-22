/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plmbase_config.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "dcm_util.h"
#include "make_string.h"

#if PLM_DCM_USE_DCMTK
#include "dcmtk_rdd.h"
#include "dcmtk_uid.h"
#elif GDCM_VERSION_1
#include "gdcm1_rdd.h"
#include "gdcm1_util.h"
#else /* GDCM_VERSION_2 */
#include "gdcm2_util.h"
#endif

void
dcm_get_date_time (
    std::string *date,
    std::string *time
)
{
#if PLM_DCM_USE_DCMTK
    *date = "20110101";
    *time = "120000";
#elif GDCM_VERSION_1
    gdcm1_get_date_time (date, time);
#else /* GDCM_VERSION_2 */
    gdcm2_get_date_time (date, time);
#endif
}

std::string 
dcm_anon_patient_id (void)
{
    int i;
    unsigned char uuid[16];
    std::string patient_id = "PL";

    srand (time (0));
    for (i = 0; i < 16; i++) {
       int r = (int) (10.0 * rand() / RAND_MAX);
       uuid[i] = '0' + r;
    }
    uuid [15] = '\0';
    patient_id = patient_id + make_string (uuid);
    return patient_id;
}

void
dcm_load_rdd (Slice_index* rdd, const char* dicom_dir)
{
#if PLM_DCM_USE_DCMTK
    dcmtk_load_rdd (rdd, dicom_dir);
#elif GDCM_VERSION_1
    gdcm1_load_rdd (rdd, dicom_dir);
#else
    /* Do nothing */
#endif
}

char*
dcm_uid (char *uid, const char *uid_root)
{
#if PLM_DCM_USE_DCMTK
    return dcmtk_uid (uid, uid_root);
#else
    return gdcm_uid (uid, uid_root);
#endif
}
