/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plmbase_config.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "plmbase.h"

#include "gdcm1_rdd.h"
#include "make_string.h"

void
dcm_get_date_time (
    std::string *date,
    std::string *time
)
{
    *date = "20110101";
    *time = "120000";
}

std::string 
dcm_anon_patient_id (void)
{
    int i;
    unsigned char uuid[16];
    std::string patient_id = "PL";

    /* Ugh.  It is a private function. */
    //    bool rc = gdcm::Util::GenerateUUID (uuid);

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
#if PLM_USE_DCMTK
    /* Do nothing */
#elif GDCM_VERSION_1
    gdcm1_load_rdd (rdd, dicom_dir);
#else
    /* Do nothing */
#endif
}
