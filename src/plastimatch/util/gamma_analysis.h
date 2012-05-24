/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _gamma_analysis_h_
#define _gamma_analysis_h_

#include "plmutil_config.h"
#include "direction_cosines.h"
#include "plm_macros.h"
#include "itk_image_type.h"

class Plm_image;

enum Gamma_output_mode {
    GAMMA,
    PASS,
    FAIL
};

/*! \brief This is the Gamma_parms class.
 * How it works is a mystery. */
class Gamma_parms {
public:
    
    float r_tol, d_tol, dose_max, gamma_max;

    Direction_cosines dc;

    Plm_image *img_in1;
    Plm_image *img_in2;
    Plm_image *img_out;

    Gamma_output_mode mode;
    bool labelmap;

public:
    Gamma_parms () {
        r_tol = d_tol = gamma_max = 3;
        mode = GAMMA;
        labelmap = false;
    }
};

/*! \brief This function finds a dose threshold.  It is a global function. 
  If we document the file (with the \file directive) or if we add this 
  function to a group, it will show up in Doxygen. */
PLMUTIL_C_API void find_dose_threshold (Gamma_parms *parms);

PLMUTIL_C_API void do_gamma_analysis (Gamma_parms *parms);

#endif
