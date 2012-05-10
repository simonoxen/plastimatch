/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _segment_body_h_
#define _segment_body_h_

#include "plmsegment_config.h"

class Plm_image;

class PLMSEGMENT_API Segment_body {
  public:
    Plm_image *img_in;
    Plm_image *img_out;

    bool m_bot_given;
    float m_bot;
    bool m_debug;
    bool m_fast;

    float m_lower_threshold;
    float m_upper_threshold;

  public:
    Segment_body () {
        m_bot_given = false;
        m_debug = false;
        m_fast = false;
        m_lower_threshold = -300;
    }
    void do_segmentation ();
    UCharImageType::Pointer threshold_patient (FloatImageType::Pointer i1);
    int find_patient_bottom (FloatImageType::Pointer i1);
};

#endif
