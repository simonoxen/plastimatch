/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plmutil_config.h"
#include <limits>

#include "float_pair_list.h"
#include "plm_math.h"

Float_pair_list
parse_float_pairs (const std::string& s)
{
    Float_pair_list al;
    const char* c = s.c_str();

    while (1) {
        int n;
        float f1, f2;
        int rc = sscanf (c, " %f , %f %n", &f1, &f2, &n);
        if (rc < 2) {
            break;
        }

        /* Look for end-caps */
        if (!is_number(f1)) {
            if (al.size() == 0) {
                f1 = -std::numeric_limits<float>::max();
            } else {
                f1 = std::numeric_limits<float>::max();
            }
        }
        /* Append (x,y) pair to list */
        al.push_back (std::make_pair (f1, f2));

        /* Look for next pair in string */
        c += n;
        if (*c == ',') c++;
    }

    return al;
}
