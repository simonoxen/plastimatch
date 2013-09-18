/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _shared_parms_h_
#define _shared_parms_h_

#include "plmregister_config.h"
#include <string>

class PLMREGISTER_API Shared_parms {
public:
    Shared_parms ();
    Shared_parms (const Shared_parms& s);
    ~Shared_parms ();
public:
    /* ROI */
    bool fixed_roi_enable;
    bool moving_roi_enable;
    std::string fixed_roi_fn;
    std::string moving_roi_fn;
public:
    void copy (const Shared_parms *s);
};

#endif