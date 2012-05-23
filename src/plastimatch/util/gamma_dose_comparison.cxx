/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plmutil_config.h"
#include <algorithm>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "itkImageRegionIteratorWithIndex.h"
#include "itkImageRegion.h"

#include "plmbase.h"
#include "plmutil.h"

#include "plm_math.h"

class Gamma_dose_comparison_private {
public:
    Gamma_parms gp;
};

Gamma_dose_comparison::Gamma_dose_comparison (...) {
    d_ptr = new Gamma_dose_comparison_private;
}

Gamma_dose_comparison::~Gamma_dose_comparison () {
    delete d_ptr;
}



#if defined (commentout)
PLMUTIL_C_API void find_dose_threshold (Gamma_parms *parms);

PLMUTIL_C_API void do_gamma_analysis (Gamma_parms *parms);
#endif
