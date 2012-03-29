/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _acquire_thread_h_
#define _acquire_thread_h_

#include <QObject>

class Dips_panel;
class Varian_4030e;

class Acquire_thread : public QObject
{
    Q_OBJECT
    ;
public slots:
    void run();
    
public:
    Acquire_thread ();
    ~Acquire_thread ();
public:
    void open_receptor (const char* path);
public:
    int idx;
    Dips_panel *dp;
    Varian_4030e *vp;
};

#endif
