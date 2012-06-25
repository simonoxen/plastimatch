/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _dice_statistics_h
#define _dice_statistics_h

#include "plmutil_config.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "plm_config.h"
#include "itkImageFileReader.h"
#include "itkImage.h"
#include "itk_image_type.h"
#include "itkImageRegionIteratorWithIndex.h"
#include "itkImageSliceConstIteratorWithIndex.h"

template<class T> float 
PLMUTIL_API
do_dice (typename itk::Image<T,3>::Pointer reference, 
    typename itk::Image<T,3>::Pointer warped, FILE* output);


template<class T> float 
PLMUTIL_API
do_dice_nsh (typename itk::Image<T,3>::Pointer reference, 
    typename itk::Image<T,3>::Pointer warped, FILE* output);


#endif
