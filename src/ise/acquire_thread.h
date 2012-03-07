/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _acquire_thread_h_
#define _acquire_thread_h_

#include <QObject>

class Acquire_thread : public QObject
{
    Q_OBJECT

    public slots:
    void run();
};

#endif
