/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plmsys_config.h"
#include <stdio.h>
#include "dlib/threads.h"
#include "dlib_threads.h"

class Dlib_thread_function_private
{
public:
    dlib::thread_function tf;
public:
    Dlib_thread_function_private (
        void (*thread_routine) (void *), void *arg) 
        : tf (thread_routine, arg)
    { }
};

Dlib_thread_function::Dlib_thread_function (
    void (*thread_routine) (void *), void *arg)
{
    d_ptr = new Dlib_thread_function_private (thread_routine, arg);
}

Dlib_thread_function::~Dlib_thread_function ()
{
    delete d_ptr;
}


class Dlib_semaphore_private
{
public:
    dlib::mutex rm;
    dlib::signaler master_rs;
    dlib::signaler slave_rs;
    bool slave_active;
    bool slave_waits;
public:
    Dlib_semaphore_private () : master_rs (rm), slave_rs (rm) {
        slave_active = false;
        slave_waits = false;
    }
};

Dlib_semaphore::Dlib_semaphore ()
{
    d_ptr = new Dlib_semaphore_private;
}

Dlib_semaphore::~Dlib_semaphore ()
{
    delete d_ptr;
}

void
Dlib_semaphore::master_grab_resource ()
 {
    d_ptr->rm.lock ();
    d_ptr->slave_waits = true;
    while (d_ptr->slave_active) {
        d_ptr->master_rs.wait ();
    }
    d_ptr->rm.unlock ();
}

void
Dlib_semaphore::master_release_resource ()
{
    d_ptr->rm.lock ();
    d_ptr->slave_waits = false;
    d_ptr->slave_rs.signal ();
    d_ptr->rm.unlock ();
}

void
Dlib_semaphore::slave_grab_resource ()
 {
    d_ptr->rm.lock ();
    while (d_ptr->slave_waits) {
        d_ptr->slave_rs.wait ();
    }
    d_ptr->slave_active = true;
    d_ptr->rm.unlock ();
}

void
Dlib_semaphore::slave_release_resource ()
{
    d_ptr->rm.lock ();
    d_ptr->slave_active = false;
    d_ptr->master_rs.signal ();
    d_ptr->rm.unlock ();
}

#include "dlib/threads/multithreaded_object_extension.cpp"
#include "dlib/threads/threaded_object_extension.cpp"
#include "dlib/threads/threads_kernel_1.cpp"
#include "dlib/threads/threads_kernel_2.cpp"
#include "dlib/threads/threads_kernel_shared.cpp"
#include "dlib/threads/thread_pool_extension.cpp"
