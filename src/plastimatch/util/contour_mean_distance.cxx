/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plm_config.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "itkImage.h"
#include "plm_ContourMeanDistanceImageFilter.h"

#include "contour_mean_distance.h"
#include "itk_image_load.h"
#include "logfile.h"

class Contour_distance_private {
public:
    Contour_distance_private () {
        mean_distance = 0.f;
    }
public:
    float mean_distance;
    UCharImageType::Pointer ref_image;
    UCharImageType::Pointer cmp_image;
};

Contour_distance::Contour_distance ()
{
    d_ptr = new Contour_distance_private;
}

Contour_distance::~Contour_distance ()
{
    delete d_ptr;
}

void 
Contour_distance::set_reference_image (const char* image_fn)
{
    d_ptr->ref_image = itk_image_load_uchar (image_fn, 0);
}

void 
Contour_distance::set_reference_image (
    const UCharImageType::Pointer image)
{
    d_ptr->ref_image = image;
}

void 
Contour_distance::set_compare_image (const char* image_fn)
{
    d_ptr->cmp_image = itk_image_load_uchar (image_fn, 0);
}

void 
Contour_distance::set_compare_image (
    const UCharImageType::Pointer image)
{
    d_ptr->cmp_image = image;
}

void 
Contour_distance::run ()
{
    typedef itk::plm_ContourMeanDistanceImageFilter< 
        UCharImageType, UCharImageType
        > ContourMeanDistanceImageFilterType;
 
    ContourMeanDistanceImageFilterType::Pointer 
        cmd_filter = ContourMeanDistanceImageFilterType::New();
    cmd_filter->SetInput1 (d_ptr->ref_image);
    cmd_filter->SetInput2 (d_ptr->cmp_image);
    cmd_filter->SetUseImageSpacing (true);
    try {
        cmd_filter->Update();
    } catch (itk::ExceptionObject &err) {
	std::cout << "ITK Exception: " << err << std::endl;
        return;
    }

    d_ptr->mean_distance 
        = cmd_filter->GetMeanDistance ();
}

float Contour_distance::get_mean_distance ()
{
    return d_ptr->mean_distance;
}

void 
Contour_distance::debug ()
{
    lprintf (
	"Contour Mean distance = %f\n",
	this->get_mean_distance ());
}

void do_contour_mean_distance (
    UCharImageType::Pointer image_1, 
    UCharImageType::Pointer image_2
)
{
    Contour_distance cd;
    cd.set_reference_image (image_1);
    cd.set_compare_image (image_2);
    cd.run ();
    cd.debug ();
}
