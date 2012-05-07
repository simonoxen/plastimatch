/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _gamma_analysis_h_
#define _gamma_analysis_h_

#include "plmutil_config.h"
#include "direction_cosines.h"
#include "plm_macros.h"

class Plm_image;

/*! \brief This is the Gamma_parms class.
 * How it works is a mystery. */
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

class Gamma_filter_private;

/*! \brief This is a sample "Filter API" class for Gamma.  
 * How it works is a mystery. */
class Gamma_filter {
public:
    Gamma_filter (...);
    ~Gamma_filter ();
public:
    Gamma_filter_private *d_ptr;
public:

    PLM_SET(char*, reference_image);
    PLM_SET(Plm_image, reference_image);
    PLM_SET(FloatImageType, reference_image);
    PLM_SET(char*, comparison_image);
    PLM_SET(Plm_image, comparison_image);
    PLM_SET(FloatImageType, comparison_image);

    PLM_GET_SET(float, s_tol);

    /*! Sets the r_tol value */
    PLM_SET(float, r_tol);
    /*! Returns the r_tol value */
    PLM_GET(float, r_tol);
};

/*! \brief This function finds a dose threshold.  It is a global function. 
  If we document the file (with the \file directive) or if we add this 
  function to a group, it will show up in Doxygen. */
C_API void find_dose_threshold (Gamma_parms *parms);

C_API void do_gamma_analysis (Gamma_parms *parms);

#endif
