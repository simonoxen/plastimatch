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

template<class T>
void do_contour_mean_dist(
    typename itk::Image<T,3>::Pointer image_1, 
    typename itk::Image<T,3>::Pointer image_2
)
{
    typedef itk::plm_ContourMeanDistanceImageFilter< 
        itk::Image<T,3> , 
        itk::Image<T,3> > ContourMeanDistanceImageFilterType;
 
    typename ContourMeanDistanceImageFilterType::Pointer 
        contourMeanDistanceImageFilter 
        = ContourMeanDistanceImageFilterType::New();
    contourMeanDistanceImageFilter->SetInput1(image_1);
    contourMeanDistanceImageFilter->SetInput2(image_2);
    contourMeanDistanceImageFilter->SetUseImageSpacing(true);
    try {
        contourMeanDistanceImageFilter->Update();
    } catch (itk::ExceptionObject &err) {
	std::cout << "ITK Exception: " << err << std::endl;
        return;
    }

    printf (
	"Contour Mean distance = %f\n",
	contourMeanDistanceImageFilter->GetMeanDistance());
}

/* Explicit instantiations */
template 
PLMUTIL_API
void do_contour_mean_dist<unsigned char> (
    itk::Image<unsigned char,3>::Pointer image_1, 
    itk::Image<unsigned char,3>::Pointer image_2);
template 
PLMUTIL_API
void do_contour_mean_dist<float> (
    itk::Image<float,3>::Pointer image_1, 
    itk::Image<float,3>::Pointer image_2);
