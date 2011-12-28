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

    /* for dose comparison plugin */
    int isodose_value1, isodose_value2, 
	isodose_value3, isodose_value4, isodose_value5;
    Plm_image *dose_labelmap1, *dose_labelmap2,
	      *dose_labelmap3, *dose_labelmap4,
	      *dose_labelmap5, *composite_labelmap;

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
plastimatch1_EXPORT void do_multi_threshold (Threshbox_parms *parms);

#endif
