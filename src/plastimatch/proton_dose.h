/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _proton_dose_h_
#define _proton_dose_h_

#include "plm_config.h"
#include "plmbase.h"
#include "proton_dose_opts.h"


#define INDEX_OF(ijk, dim) \
    (((ijk[2] * dim[1] + ijk[1]) * dim[0]) + ijk[0])


#if defined __cplusplus
extern "C" {
#endif

gpuit_EXPORT 
void
proton_dose_compute (
    Volume* dose_vol,
    Volume* ct_vol,
    Proton_dose_options* options
);

#if defined __cplusplus
}
#endif

#endif
