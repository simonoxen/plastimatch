/* -------------------------------------------------------------------------*
    See COPYRIGHT for copyright information.
 * -------------------------------------------------------------------------*/
#ifndef __iqt_tracker_h__
#define __iqt_tracker_h__

class Tracker_thread;

class Tracker {
public:
    Tracker ();
    ~Tracker ();
public:
    Tracker_thread *tracker_thread;
};

#endif
