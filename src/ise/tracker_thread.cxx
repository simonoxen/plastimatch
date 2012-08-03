/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
/* Libraries */
#include <math.h>
#include <QDebug>
#include <QMutex>
#include <QWaitCondition>
#include <stdlib.h>
#include <string.h>

/* Headers */
#include "fatm.h"
#include "iqt_application.h"
#include "iqt_tracker.h"
#include "sleeper.h"
#include "tracker_thread.h"

Tracker_thread::Tracker_thread ()
{
}

Tracker_thread::~Tracker_thread () {
}

void 
Tracker_thread::set_tracker (Tracker *t)
{
    this->tracker = t;
}

void
Tracker_thread::run ()
{
    while (1) {
	ise_app->mutex.lock();
	ise_app->frameLoaded.wait(&(ise_app->mutex));
	ise_app->frameLoaded.wakeAll();

	this->tracker->tracker_initialize();
	fatm_run(tracker->fatm);

        /* Find max */
#if defined (commentout)
        double best_score = -1;
        double best_row = 0;
        double best_col = 0;
        for (r = rows; ...) {
            for (c = cols; ...) {
                double score = image_data(tracker->fatm->score)[r*175 + c];
                if (score > best_score) {
                    best_score = score;
                    best_row = r;
                    best_col = c;
                }
            }
        }
#endif

#if defined (commentout)
        static int i = 0;
        QString fn;
        fn.sprintf ("/PHShome/gcs6/igrt_research/tmp/score_%04d.pfm", i);
        image_write (&tracker->fatm->score, fn.toUtf8().constData());
        fn.sprintf ("/PHShome/gcs6/igrt_research/tmp/sig_%04d.pfm", i);
        image_write (&tracker->fatm->sig, fn.toUtf8().constData());
        fn.sprintf ("/PHShome/gcs6/igrt_research/tmp/pat_%04d.pfm", i);
        image_write (&tracker->fatm->pat, fn.toUtf8().constData());

        i++;
#endif
	ise_app->mutex.unlock();
    }
    /* after running, send signal to this->tracker->mw with scoring information */
    /* or to this->tracker->mw->vid_screen for the point around which to make
       the tracking square */
}
