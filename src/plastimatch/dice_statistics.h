/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _contour_statistics_h
#define _contour_statistics_h

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "plm_config.h"
#include "itkImageFileReader.h"
#include "itkImage.h"
#include "itk_image.h"
#include "itkImageRegionIteratorWithIndex.h"
#include "itkImageSliceConstIteratorWithIndex.h"

template<class T> float 
do_dice (typename itk::Image<T,3>::Pointer reference, 
    typename itk::Image<T,3>::Pointer warped, FILE* output);


template<class T> float 
do_dice_nsh (typename itk::Image<T,3>::Pointer reference, 
    typename itk::Image<T,3>::Pointer warped, FILE* output);


#endif
