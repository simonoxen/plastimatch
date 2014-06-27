/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _dlib_threads_h_
#define _dlib_threads_h_

#include "plmsys_config.h"

class Dlib_thread_function_private;
class Dlib_semaphore_private;

class PLMSYS_API Dlib_thread_function
{
public:
    Dlib_thread_function_private *d_ptr;
public:
    Dlib_thread_function (void (*thread_routine) (void *), void *arg);
    ~Dlib_thread_function ();
};

class PLMSYS_API Dlib_semaphore
{
public:
    Dlib_semaphore_private *d_ptr;
public:
    Dlib_semaphore ();
    ~Dlib_semaphore ();
public:
    void grab_semaphore ();
    void release_semaphore ();
};

#endif
