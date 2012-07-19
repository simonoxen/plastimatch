/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <QDebug>
#include <QMutex>
#include <QWaitCondition>

#include "iqt_tracker.h"
#include "tracker_thread.h"
#include "iqt_application.h"
#include "sleeper.h"

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
    qDebug ("****************** TRACKER STARTED **********************");
    while (1) {
	ise_app->mutex.lock();
	ise_app->frameLoaded.wait(&(ise_app->mutex));
	qDebug() << "Tracker thread called";
	ise_app->frameLoaded.wakeAll();
	ise_app->mutex.unlock();
    }
}
