/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "plm_config.h"
#include "itkHausdorffDistanceImageFilter.h"
#include "itkImage.h"

#include "itk_image.h"
#include "hausdorff_statistics.h"

template<class T>
void do_hausdorff(
    typename itk::Image<T,3>::Pointer image_1, 
    typename itk::Image<T,3>::Pointer image_2)
{
    typedef itk::HausdorffDistanceImageFilter< itk::Image<T,3>, itk::Image<T,3> > Hausdorff_filter;
    typename Hausdorff_filter::Pointer h_filter = Hausdorff_filter::New ();
    h_filter->SetInput1 (image_1);
    h_filter->SetInput2 (image_2);
    h_filter->Update ();

    //Hausdorff_filter::Pointer out = h_filter->GetOutput();

    printf (
	"Hausdorff distance = %f\n"
	"Average Hausdorff distance = %f\n",
	h_filter->GetHausdorffDistance (),
	h_filter->GetAverageHausdorffDistance ());
}

template 
void do_hausdorff<unsigned char> (
    itk::Image<unsigned char,3>::Pointer image_1, 
    itk::Image<unsigned char,3>::Pointer image_2);
template 
void do_hausdorff<float> (
    itk::Image<float,3>::Pointer image_1, 
    itk::Image<float,3>::Pointer image_2);
