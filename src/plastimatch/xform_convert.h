/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _xform_convert_h_
#define _xform_convert_h_

#include "plm_config.h"
#include "xform.h"

class plastimatch1_EXPORT Xform_convert {
public:
    Xform *xf_out;
    Xform *xf_in;
    XFormInternalType xf_out_type;
    float origin[3];
    float spacing[3];
    int dim[3];
    float grid_spac[3];
    int nobulk;
public:
    Xform_convert () {
	/* Pretty much zeros are good all around */
    }
};

plastimatch1_EXPORT
void xform_convert (Xform_convert *xfc);

#endif
