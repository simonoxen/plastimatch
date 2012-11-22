/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _contour_mean_distance_h_
#define _contour_mean_distance_h_

#include "plmutil_config.h"
#include "itkImage.h"

template<class T>
PLMUTIL_API
void do_contour_mean_dist(
    typename itk::Image<T,3>::Pointer image_1, 
    typename itk::Image<T,3>::Pointer image_2);

#endif
