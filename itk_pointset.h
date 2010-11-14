/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _itk_pointset_h_
#define _itk_pointset_h_

#include "plm_config.h"
#include "itkPointSet.h"

#include "itk_image.h"
#include "pointset.h"
#include "xform.h"

typedef itk::PointSet< float, 3 > FloatPointSetType;
typedef FloatPointSetType::PointIdentifier FloatPointIdType;
typedef itk::PointSet< double, 3 > DoublePointSetType;
typedef DoublePointSetType::PointIdentifier DoublePointIdType;

template<class T> void itk_pointset_load (T pointset, const char* fn);
template<class T> T itk_pointset_warp (T ps_in, Xform* xf);
template<class T> void itk_pointset_debug (T pointset);
FloatPointSetType::Pointer itk_float_pointset_from_pointset (Pointset *ps);
DoublePointSetType::Pointer itk_double_pointset_from_pointset (Pointset *ps);

#endif
