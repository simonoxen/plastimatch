/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _itk_pointset_h_
#define _itk_pointset_h_

#include "itk_image.h"
#include "xform.h"
#include "itkPointSet.h"

typedef itk::PointSet< float, 3 > PointSetType;

template<class T> void pointset_load (T pointset, char* fn);
template<class T> T pointset_warp (T ps_in, Xform* xf);
template<class T> void pointset_debug (T pointset);

#endif
