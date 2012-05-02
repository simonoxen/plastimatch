/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _interpolate_h_
#define _interpolate_h_

/**
*  You probably do not want to #include this header directly.
 *
 *   Instead, it is preferred to #include "plmbase.h"
 */

#include "plmbase_config.h"
#include "sys/plm_int.h"

class Volume;

C_API void li_clamp (
        float ma,
        plm_long dmax,
        plm_long* maf,
        plm_long* mar, 
        float* fa1, float* fa2
);
C_API void li_clamp_3d (
        float mijk[3],
        plm_long mijk_f[3],
        plm_long mijk_r[3],
        float li_frac_1[3],
        float li_frac_2[3],
        Volume *moving
);
C_API float li_value (
        float fx1, float fx2,
        float fy1, float fy2, 
        float fz1, float fz2,
        plm_long mvf, 
        float *m_img,
        Volume *moving
);

#endif
