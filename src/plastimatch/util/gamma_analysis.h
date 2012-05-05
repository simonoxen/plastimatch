/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _gamma_analysis_h_
#define _gamma_analysis_h_

#include "plmutil_config.h"
#include "direction_cosines.h"

class Plm_image;

class Gamma_parms {
public:
    
    float r_tol, d_tol, dose_max, gamma_max;

    Direction_cosines dc;

    Plm_image *img_in1;
    Plm_image *img_in2;
    Plm_image *img_out;
    Plm_image *img_out_pass;
    Plm_image *img_out_fail;
	Plm_image *labelmap_out;

public:
    Gamma_parms () {

    r_tol = d_tol = gamma_max = 3;

    }
};

C_API void find_dose_threshold (Gamma_parms *parms);
C_API void do_gamma_analysis (Gamma_parms *parms);

#endif
