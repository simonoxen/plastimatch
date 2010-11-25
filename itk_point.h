/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _itk_point_h_
#define _itk_point_h_

#include "plm_config.h"
#include "itkPoint.h"
#include "itkVector.h"

/* Points & vectors */
typedef itk::Point < float, 3 > FloatPointType;
typedef itk::Point < double, 3 > DoublePointType;

typedef itk::Vector < float, 3 > FloatVectorType;
typedef itk::Vector < double, 3 > DoubleVectorType;

#endif
