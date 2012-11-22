/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plm_config.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "itkImage.h"
#include "plm_HausdorffDistanceImageFilter.h"

#include "hausdorff_distance.h"
#include "itk_image_load.h"

class Hausdorff_distance_private {
public:
    Hausdorff_distance_private () {
        avg_hausdorff_distance = 0.f;
        hausdorff_distance = 0.f;
    }
public:
    float avg_hausdorff_distance;
    float hausdorff_distance;
    UCharImageType::Pointer ref_image;
    UCharImageType::Pointer cmp_image;
};

Hausdorff_distance::Hausdorff_distance ()
{
    d_ptr = new Hausdorff_distance_private;
}

Hausdorff_distance::~Hausdorff_distance ()
{
    delete d_ptr;
}

void 
Hausdorff_distance::set_reference_image (const char* image_fn)
{
    d_ptr->ref_image = itk_image_load_uchar (image_fn, 0);
}

void 
Hausdorff_distance::set_reference_image (
    const UCharImageType::Pointer image)
{
    d_ptr->ref_image = image;
}

void 
Hausdorff_distance::set_compare_image (const char* image_fn)
{
    d_ptr->cmp_image = itk_image_load_uchar (image_fn, 0);
}

void 
Hausdorff_distance::set_compare_image (
    const UCharImageType::Pointer image)
{
    d_ptr->cmp_image = image;
}

void 
Hausdorff_distance::run ()
{
    typedef unsigned char T;
    typedef itk::plm_HausdorffDistanceImageFilter< 
	itk::Image<T,3>, itk::Image<T,3> > Hausdorff_filter;
//    typename Hausdorff_filter::Pointer h_filter = Hausdorff_filter::New ();
    Hausdorff_filter::Pointer h_filter = Hausdorff_filter::New ();

    h_filter->SetInput1 (d_ptr->ref_image);
    h_filter->SetInput2 (d_ptr->cmp_image);
    h_filter->SetUseImageSpacing (true);
    try {
        h_filter->Update ();
    } catch (itk::ExceptionObject &err) {
	std::cout << "ITK Exception: " << err << std::endl;
        return;
    }
    d_ptr->hausdorff_distance 
        = h_filter->GetHausdorffDistance ();
    d_ptr->avg_hausdorff_distance 
        = h_filter->GetAverageHausdorffDistance ();

}

float Hausdorff_distance::get_hausdorff ()
{
    return d_ptr->hausdorff_distance;
}

float Hausdorff_distance::get_average_hausdorff ()
{
    return d_ptr->avg_hausdorff_distance;
}

void 
do_hausdorff (
    UCharImageType::Pointer image_1, 
    UCharImageType::Pointer image_2
)
{
    Hausdorff_distance hd;
    hd.set_reference_image (image_1);
    hd.set_compare_image (image_2);
    hd.run();
    printf (
	"Hausdorff distance = %f\n"
	"Average Hausdorff distance = %f\n",
	hd.get_hausdorff (),
	hd.get_average_hausdorff ());
}
