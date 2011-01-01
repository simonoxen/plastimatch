/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _segment_body_h_
#define _segment_body_h_

#include "plm_config.h"
#include "plm_image.h"

class Segment_body {
 public:
    Plm_image img_in;
    Plm_image img_out;

    bool bot_given;
    float bot;

 public:
    plastimatch1_EXPORT 
	void do_segmentation ();
};

#endif
