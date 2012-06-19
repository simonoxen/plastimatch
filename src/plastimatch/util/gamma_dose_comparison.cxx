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
    Gamma_dose_comparison_private () {
        have_reference_dose = false;
        have_gamma_image = false;
    }
public:
    Gamma_parms gp;
    bool have_reference_dose;
    bool have_gamma_image;
};

Gamma_dose_comparison::Gamma_dose_comparison () {
    d_ptr = new Gamma_dose_comparison_private;
}

Gamma_dose_comparison::~Gamma_dose_comparison () {
    delete d_ptr;
}

void 
Gamma_dose_comparison::set_reference_image (const char* image_fn)
{
    d_ptr->gp.img_in1 = new Plm_image (image_fn);
}

void 
Gamma_dose_comparison::set_reference_image (Plm_image* image)
{
    d_ptr->gp.img_in1 = image;
}

void 
Gamma_dose_comparison::set_reference_image (
    const FloatImageType::Pointer image)
{
    d_ptr->gp.img_in1 = new Plm_image (image);
}

void 
Gamma_dose_comparison::set_compare_image (const char* image_fn)
{
    d_ptr->gp.img_in2 = new Plm_image (image_fn);
}

void 
Gamma_dose_comparison::set_compare_image (Plm_image* image)
{
    d_ptr->gp.img_in2 = image;
}

void 
Gamma_dose_comparison::set_compare_image (
    const FloatImageType::Pointer image)
{
    d_ptr->gp.img_in2 = new Plm_image (image);
}

float
Gamma_dose_comparison::get_spatial_tolerance ()
{
    return d_ptr->gp.r_tol;
}

void 
Gamma_dose_comparison::set_spatial_tolerance (float spatial_tol)
{
    d_ptr->gp.r_tol = spatial_tol;
}

void 
Gamma_dose_comparison::set_dose_difference_tolerance (float dose_tol)
{
    d_ptr->gp.d_tol = dose_tol;
}

void 
Gamma_dose_comparison::set_reference_dose (float dose)
{
    d_ptr->gp.dose_max = dose;
    d_ptr->have_reference_dose = true;
}

void 
Gamma_dose_comparison::set_gamma_max (float gamma_max)
{
    d_ptr->gp.gamma_max = gamma_max;
}

void 
Gamma_dose_comparison::run ()
{
    if (!d_ptr->have_reference_dose) {
        find_dose_threshold (&d_ptr->gp);
    }
    d_ptr->have_gamma_image = true;
    do_gamma_analysis (&d_ptr->gp);
}

Plm_image*
Gamma_dose_comparison::get_gamma_image ()
{
    if (!d_ptr->have_gamma_image) {
        this->run();
    }
    return d_ptr->gp.img_out;
}

FloatImageType::Pointer
Gamma_dose_comparison::get_gamma_image_itk ()
{
    return get_gamma_image()->itk_float();
}

Plm_image*
Gamma_dose_comparison::get_pass_image ()
{
    if (!d_ptr->have_gamma_image) {
        this->run();
    }
    d_ptr->gp.mode = PASS;
    do_gamma_threshold (&d_ptr->gp);
    return d_ptr->gp.labelmap_out;
}

UCharImageType::Pointer
Gamma_dose_comparison::get_pass_image_itk ()
{
    return get_pass_image()->itk_uchar();
}

Plm_image*
Gamma_dose_comparison::get_fail_image ()
{
    if (!d_ptr->have_gamma_image) {
        this->run();
    }
    d_ptr->gp.mode = FAIL;
    do_gamma_threshold (&d_ptr->gp);
    return d_ptr->gp.labelmap_out;
}

UCharImageType::Pointer
Gamma_dose_comparison::get_fail_image_itk ()
{
    return get_fail_image()->itk_uchar();
}
