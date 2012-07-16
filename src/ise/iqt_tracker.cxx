/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "fatm.h"
#include "image.h"
#include "iqt_tracker.h"
#include "tracker_thread.h"

Tracker::Tracker () 
{
    this->tracker_thread = new Tracker_thread;
    //this->tracker_thread->start(); 
}

Tracker::~Tracker () 
{
    delete tracker_thread;
}


void
tracker_initialize (/*bool user_track*/)
{
#if defined (this_is_today)
    FATM_Options *fopt;
    int pat_size[] = { 21, 21 };
    int sig_size[] = { 151, 151 };

    /* Initialize fopt array */
    fopt = fatm_initialize ();

    /* 
       fopt->pat_rect = pointer to pattern
       fopt->sig_rect = pointer to signal 
       fopt->score = pointer to score
    */

    fopt->command = MATCH_COMMAND_COMPILE;
    fopt->alg = MATCH_ALGORITHM_FNCC;
    fatm_compile (fopt);

    fatm_run (fopt);
    /*
      fopt->pat_rect = new pattern 
      fatm_run (fopt);
    */
#endif

#if defined (this_is_what_we_would_like)
    Fatm fatm = new Fatm;
    Image_Rect *pattern;
    Image_Rect *signal;
    int dims[2] = {10, 10};
    pattern->set_dims (dims);
    signal->set_dims ({150, 150}); /*arbitrary, could be user-defined as well*/
    unsigned short *img = pattern->get_image_pointer(); /* will be same as f->img */
    
    for (int i = 0; i < signal->dims[0]; i++) {
	for (int j = 0; j < signal->dims[1]; j++) {
	    signal->data[i*signal->dims[0] + j] = img[i*(512) + j];
	}
    }

    if (!user_track) {
	//unsigned short value;
	int value;
	
	/* create pattern */
	for (int i = 0; i < dims[0]; i++) {
	    for (int j = 0; j < dims[1]; j++) {
		if (i==4 || i==5) {
		    if (j > 1 && j < 8) {
			value = 0xFFFF;
		    } else {
			value = 0;
		    }
		} else {
		    value = 0;
		}
		/* horizontal bright rectangle */
		pattern->data[i*dims[1]+j] = value;
	    }
	}
    }
    
    else {
	for (int i=0; i < 10; i++) {
	    for (int j=0; j < 10; j++) {
		pattern->data[i*dims[1]+j] = img[(pattern->pmin[0]+i)*(512) 
						 + (pattern->pmin[1]+j)];
		/* pmin[] is top-left pixel ({x, y}) of user-defined pattern,
		   would be from "trace" in iqt_video_widget 
		   need to know how the img[] array is laid out */
	    }
	}
    }    

    fatm->set_pattern (pattern);
#endif
}
