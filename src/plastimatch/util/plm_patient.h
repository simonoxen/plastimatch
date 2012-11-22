/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _plm_patient_h_
#define _plm_patient_h_

#include "plmutil_config.h"

class Plm_patient_private;

class PLMUTIL_API Plm_patient 
{
public:
    Plm_patient ();
    ~Plm_patient ();

public:
    Plm_patient_private *d_ptr;

public:
    void load_rdd (const char *rdd);
    void debug (void) const;
};

#endif
