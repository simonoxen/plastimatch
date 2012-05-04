/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _xform_convert_h_
#define _xform_convert_h_

#include "plmbase_config.h"
#include "volume_header.h"

#include "xform.h"  /* cannot forward declare enum in C++ */

class plastimatch1_EXPORT Xform_convert {
public:
    Xform *m_xf_out;
    Xform *m_xf_in;
    XFormInternalType m_xf_out_type;
    Volume_header m_volume_header;
    float m_grid_spac[3];
    int m_nobulk;
public:
    Xform_convert () {
	m_xf_out = 0;
	m_xf_in = 0;
	m_xf_out_type = XFORM_NONE;

	for (int d = 0; d < 3; d++) {
	    m_grid_spac[d] = 100.f;
	}
	m_nobulk = false;
    }
    ~Xform_convert () {
	if (m_xf_out) delete m_xf_out;
	if (m_xf_in) delete m_xf_in;
    }
};

plastimatch1_EXPORT
void xform_convert (Xform_convert *xfc);

#endif
