/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _rtds_h_
#define _rtds_h_

#include "plm_config.h"
#include "cxt_io.h"
#include "plm_image.h"

/* rtds = RT data set */
class Rtds {
public:
    PlmImage *m_img;
    Cxt_structure_list *m_cxt;
    PlmImage *m_dose;
public:
    Rtds () {
	m_img = 0;
	m_cxt = 0;
	m_dose = 0;
    }
    ~Rtds () {
	if (m_img) {
	    delete m_img;
	}
	if (m_cxt) {
	    cxt_destroy (m_cxt);
	}
	if (m_dose) {
	    delete m_img;
	}
    }
    void load_xio (char *xio_dir);
};

#if defined __cplusplus
extern "C" {
#endif


#if defined __cplusplus
}
#endif

#endif
