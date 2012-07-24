/* -------------------------------------------------------------------------*
    See COPYRIGHT for copyright information.
 * -------------------------------------------------------------------------*/
#ifndef _iqt_tracker_h_
#define _iqt_tracker_h_

class Tracker_thread;

class Tracker {
public:
    Tracker ();
    ~Tracker ();
public:
    Tracker_thread *tracker_thread;
    void tracker_initialize ();
};

#endif
