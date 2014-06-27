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
    dlib::signaler rs;
    bool semaphore_available;
public:
    Dlib_semaphore_private () : rs (rm) {
        semaphore_available = true;
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
Dlib_semaphore::grab_semaphore ()
{
    d_ptr->rm.lock ();
    while (d_ptr->semaphore_available == false) {
        d_ptr->rs.wait ();
    }
    d_ptr->semaphore_available = false;
    d_ptr->rm.unlock ();
}

void
Dlib_semaphore::release_semaphore ()
{
    d_ptr->rm.lock ();
    d_ptr->semaphore_available = true;
    d_ptr->rm.unlock ();
    d_ptr->rs.signal ();
}

#include "dlib/threads/multithreaded_object_extension.cpp"
#include "dlib/threads/threaded_object_extension.cpp"
#include "dlib/threads/threads_kernel_1.cpp"
#include "dlib/threads/threads_kernel_2.cpp"
#include "dlib/threads/threads_kernel_shared.cpp"
#include "dlib/threads/thread_pool_extension.cpp"
