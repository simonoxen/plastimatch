/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plmbase_config.h"
#include <string>
#include <stdio.h>
#include <stdlib.h>

#include "dcmtk_util.h"

void
dcmtk_get_date_time (
    std::string *date,
    std::string *time
)
{
//        DcmDate::getCurrentDate (date_string);
//        DcmTime::getCurrentTime (time_string);
    *date = "20110101";
    *time = "120000";
}
