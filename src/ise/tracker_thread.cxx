/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <QDebug>

#include "iqt_tracker.h"
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
Tracker_thread::run () {
    qDebug() << "Hello from tracker_thread!!!";
}
