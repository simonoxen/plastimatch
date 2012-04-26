/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _mc_dose_h_
#define _mc_dose_h_

#include "plmbase_config.h"
#include "plm_image.h"
#include "xio_ct.h"

plastimatch1_EXPORT 
void
mc_dose_load (Plm_image *plm, const char *filename);
plastimatch1_EXPORT 
void
mc_dose_apply_transform (Plm_image *plm, Xio_ct_transform *transform);

#endif
