/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _diff_h_
#define _diff_h_

#include "plmutil_config.h"
#include "pstring.h"

class PLMUTIL_API Diff_parms {
public:
    Pstring img_in_1_fn;
    Pstring img_in_2_fn;
    Pstring img_out_fn;
public:
    Diff_parms ();
};

PLMUTIL_API void diff_main (Diff_parms* parms);

#endif
