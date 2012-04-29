/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plm_config.h"
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
    *date = "20110101";
    *time = "120000";
}
