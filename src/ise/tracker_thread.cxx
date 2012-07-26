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
	ise_app->mutex.unlock();
	fatm_run(tracker->fatm);
    }
    /* after running, send signal to this->tracker->mw with scoring information */
}
