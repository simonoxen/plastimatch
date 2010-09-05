/*===========================================================
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
===========================================================*/
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
//#include "slice_extract.h"


/* =======================================================================*
    Definitions
 * =======================================================================*/
//typedef UCharImageType ImgType;
//typedef itk::Image<unsigned char, 2>	intImgType;



//void do_dice_global(ImgType::Pointer reference, ImgType::Pointer warped, FILE* output);
//void do_dice_expert(ImgType::Pointer ex_1, ImgType::Pointer ex_2, ImgType::Pointer ex_3, FILE* output);
template<class T> float do_dice_global(typename itk::Image<T,Dimension>::Pointer reference, 
									   typename itk::Image<T,Dimension>::Pointer warped, FILE* output, T);

#endif
