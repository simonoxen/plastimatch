/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _gamma_dose_comparison_h_
#define _gamma_dose_comparison_h_

#include "plmutil_config.h"
#include "direction_cosines.h"
#include "plm_macros.h"
#include "itk_image_type.h"

class Gamma_dose_comparison_private;
class Plm_image;

/*! \file This is gamma_dose_comparison.h */

/*! \brief This is a sample "Filter PLMUTIL_API" class for Gamma.  
 * How it works is a mystery. */
class Gamma_dose_comparison {
public:
    Gamma_dose_comparison (...);
    ~Gamma_dose_comparison ();
public:
    Gamma_dose_comparison_private *d_ptr;
public:

    /*! \name Parameter setting */
    ///@{
    PLM_SET(const char*, reference_image);
    PLM_SET_CR(Plm_image, reference_image);
    PLM_SET_CR(FloatImageType, reference_image);
    PLM_SET(const char*, comparison_image);
    PLM_SET_CR(Plm_image, comparison_image);
    PLM_SET_CR(FloatImageType, comparison_image);
    ///@}

    /*! \name Execution */
    ///@{
    void run ();
    ///@}

    /*! \name Getting outputs */
    ///@{
    Plm_image* get_gamma_img ();
    FloatImageType::Pointer get_gamma_img_itk ();
    Plm_image* get_pass_img ();
    UCharImageType::Pointer get_pass_img_itk ();
    Plm_image* get_fail_img ();
    UCharImageType::Pointer get_fail_img_itk ();
    Plm_image* get_labelmap_pass_img ();
    UCharImageType::Pointer get_labelmap_fail_img_itk ();
    ///@}
};

#endif
