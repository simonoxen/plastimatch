/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plmsys_config.h"

#include "bstring_util.h"

bool
bstring_empty (const CBString& cbstring)
{
    return cbstring.length() == 0;
}

bool
bstring_not_empty (const CBString& cbstring)
{
    return cbstring.length() != 0;
}
