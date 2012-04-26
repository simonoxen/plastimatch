/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "plm_config.h"
#include "plm_HausdorffDistanceImageFilter.h"
#include "itkImage.h"

#include "itk_image.h"
#include "hausdorff_statistics.h"
#include "plm_ContourMeanDistanceImageFilter.h"

template<class T>
void do_hausdorff (
    typename itk::Image<T,3>::Pointer image_1, 
    typename itk::Image<T,3>::Pointer image_2
)
{
    typedef itk::plm_HausdorffDistanceImageFilter< 
	itk::Image<T,3>, itk::Image<T,3> > Hausdorff_filter;
    typename Hausdorff_filter::Pointer h_filter = Hausdorff_filter::New ();
    h_filter->SetInput1 (image_1);
    h_filter->SetInput2 (image_2);
    h_filter->SetUseImageSpacing(true);
    h_filter->Update ();

    printf (
	"Hausdorff distance = %f\n"
	"Average Hausdorff distance = %f\n",
	h_filter->GetHausdorffDistance (),
	h_filter->GetAverageHausdorffDistance ());
}

template<class T>
void do_contour_mean_dist(
    typename itk::Image<T,3>::Pointer image_1, 
    typename itk::Image<T,3>::Pointer image_2
)
{

  typedef itk::plm_ContourMeanDistanceImageFilter< itk::Image<T,3> , itk::Image<T,3>  > ContourMeanDistanceImageFilterType;
 
  typename ContourMeanDistanceImageFilterType::Pointer contourMeanDistanceImageFilter = ContourMeanDistanceImageFilterType::New();
  contourMeanDistanceImageFilter->SetInput1(image_1);
  contourMeanDistanceImageFilter->SetInput2(image_2);
  contourMeanDistanceImageFilter->SetUseImageSpacing(true);
  contourMeanDistanceImageFilter->Update();
 
  printf (
	"Contour Mean distance = %f\n",
	contourMeanDistanceImageFilter->GetMeanDistance());

}

/* Explicit instantiations */
template 
void do_hausdorff<unsigned char> (
    itk::Image<unsigned char,3>::Pointer image_1, 
    itk::Image<unsigned char,3>::Pointer image_2);
template 
void do_hausdorff<float> (
    itk::Image<float,3>::Pointer image_1, 
    itk::Image<float,3>::Pointer image_2);

template 
void do_contour_mean_dist<unsigned char> (
    itk::Image<unsigned char,3>::Pointer image_1, 
    itk::Image<unsigned char,3>::Pointer image_2);
template 
void do_contour_mean_dist<float> (
    itk::Image<float,3>::Pointer image_1, 
    itk::Image<float,3>::Pointer image_2);
