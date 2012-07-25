/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "fatm.h"
#include "iqt_tracker.h"
#include "tracker_thread.h"
#include "iqt_main_window.h"
#include "frame.h"
#include "iqt_application.h"
#include "cbuf.h"
#include <QDebug>

Tracker::Tracker (Iqt_main_window *mw) 
{
    this->tracker_thread = new Tracker_thread;
    this->tracker_thread->set_tracker (this);
    this->mw = mw;
}

Tracker::~Tracker () 
{
    delete tracker_thread;
}


void
Tracker::tracker_initialize (/*bool user_track*/)
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

    bool user_track = false;
    bool user_def = false;
    FATM_Options *fatm = new FATM_Options;
    Image_Rect *pattern;
    Image_Rect *signal;
    int dims[2] = {10, 10};
    qDebug("dims: %d, %d", dims[0], dims[1]);
    int sig_dims[2] = {175, 175};
    pattern->set_dims (dims);
    //signal->set_dims (sig_dims); /*arbitrary, could be user-defined as well*/
    
    fatm = fatm_initialize ();
    fatm->alg = MATCH_ALGORITHM_FNCC;
        
    Frame *f = *(ise_app->cbuf[0]->display_ptr);
    unsigned short *img = f->img;
    
    if (!user_def) {
	signal->set_dims (sig_dims);
	signal->pmin[0] = 175;
	signal->pmin[1] = 175;
    }
    
    for (int i = signal->pmin[0]; i < (signal->dims[0] + signal->pmin[0]); i++) {
	for (int j = signal->pmin[1]; j < (signal->dims[1] + signal->pmin[1]); j++){
	    signal->data[i*signal->dims[0] + j] = img[i*(512) + j];
	}
    }

    fatm->sig_rect = *(signal);

    if (!user_track) {
	unsigned short value;
	//int value;
	
	/* create pattern */
	for (int i = 0; i < dims[0]; i++) {
	    for (int j = 0; j < dims[1]; j++) {
		if (i>1 && i<8) {
		    if (j == 4 || j == 5) {
			value = 0xFFFF;
		    } else {
			value = 0;
		    }
		} else {
		    value = 0;
		}
		/* vertical bright rectangle */
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

    fatm->pat_rect = *(pattern);
    fatm_compile (fatm);
}
