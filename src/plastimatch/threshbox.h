/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _threshbox_h_
#define _threshbox_h_

#include "plm_config.h"
#include "direction_cosines.h"
#include "itk_image.h"
#include "plm_image.h"

class Threshbox_parms {
public:
    int center[3];
    int boxsize[3];
    int threshold;

    Direction_cosines dc;

    Plm_image *img_in;
    Plm_image *img_out;
    Plm_image *img_box;

    Plm_image *overlap_labelmap1;
    Plm_image *overlap_labelmap2;
    char overlap_fn[1024];

public:
    Threshbox_parms () {
    center[0]=100;
    center[1]=100;
    center[3]=100;
    boxsize[0]=10;
    boxsize[1]=10;
    boxsize[2]=10;
    threshold=80;
    
    }
};

plastimatch1_EXPORT void do_threshbox (Threshbox_parms *parms);
plastimatch1_EXPORT void do_overlap_fraction (Threshbox_parms *parms);


#endif
