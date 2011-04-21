/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _segment_body_h_
#define _segment_body_h_

#include "plm_config.h"
#include "plm_image.h"

class plastimatch1_EXPORT Segment_body {
  public:
    Plm_image img_in;
    Plm_image img_out;

    bool m_bot_given;
    float m_bot;
    bool m_debug;
    bool m_fast;

  public:
    Segment_body () {
	m_bot_given = false;
	m_debug = false;
	m_fast = false;
    }
    void do_segmentation ();
};

#endif
