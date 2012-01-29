/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _autolabel_h_
#define _autolabel_h_

#include "plm_config.h"
#include "pstring.h"

class Autolabel_parms {
public:
    Autolabel_parms () {
	enforce_anatomic_constraints = false;
    }
public:
    Pstring input_fn;
    Pstring output_csv_fn;
    Pstring output_fcsv_fn;
    Pstring network_fn;
    Pstring task;
    bool enforce_anatomic_constraints;
};

void autolabel (Autolabel_parms *parms);

#endif
