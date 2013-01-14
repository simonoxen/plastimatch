/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plmsys_config.h"
#include <stdio.h>

#include "option_range.h"
#include "plm_math.h"
#include "string_util.h"

class Option_range_private {
public:
    std::list<float> range;
};

Option_range::Option_range ()
{
    d_ptr = new Option_range_private;
}

Option_range::~Option_range ()
{
    delete d_ptr;
}

void
Option_range::set_log_range (const std::string& range)
{
    float min_value;
    float max_value;
    float incr;
    int rc = sscanf (range.c_str(), "%f:%f:%f", &min_value, &incr, &max_value);
    if (rc != 3) {
        return;
    }

    /* Create list of values */
    d_ptr->range.clear ();
    for (float log_value = min_value; 
        log_value <= max_value;
        log_value += incr)
    {
        d_ptr->range.push_back (exp10_ (log_value));
    }
}

void
Option_range::set_linear_range (const std::string& range)
{
    float min_value;
    float max_value;
    float incr;
    int rc = sscanf (range.c_str(), "%f:%f:%f", &min_value, &incr, &max_value);
    if (rc != 3) {
        return;
    }

    /* Create list of values */
    d_ptr->range.clear ();
    for (float lin_value = min_value; 
         lin_value <= max_value;
         lin_value += incr)
    {
        d_ptr->range.push_back (lin_value);
    }
}

const std::list<float>&
Option_range::get_range ()
{
    return d_ptr->range;
}
