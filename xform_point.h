/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _xform_point_h_
#define _xform_point_h_

#include "plm_config.h"

#include "xform.h"

plastimatch1_EXPORT void xform_point_transform (FloatPoint3DType* point_out, Xform* xf_in, FloatPoint3DType point_in);

#endif
