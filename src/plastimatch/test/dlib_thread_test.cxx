#include <stdio.h>
#include "dlib_threads.h"

#if _WIN32
#include <windows.h>
#define plm_sleep(x) Sleep(x)
#else
#include <unistd.h>
#define plm_sleep(x) usleep(1000*x)
#endif


bool time_to_die = false;

void thread_func (void* param)
{
    Dlib_semaphore *s = (Dlib_semaphore *) param;
    
    while (1) {
        s->grab_semaphore ();
        plm_sleep (300);
        printf ("Child execute\n");
        s->release_semaphore ();
        if (time_to_die) {
            break;
        }
    }
}

int main ()
{
    Dlib_semaphore s;

    Dlib_thread_function tf (thread_func, &s);

    /* Parent and child execute simultaneously */
    for (int i = 0; i < 3; i++) {
        plm_sleep (770);
        printf ("Parent execute\n");
    }

    /* Only parent executes */
    s.grab_semaphore ();
    printf (">>> Parent only\n");
    for (int i = 0; i < 15; i++) {
        plm_sleep (70);
        printf ("Parent execute\n");
    }
    printf (">>> End parent only\n");
    s.release_semaphore ();

    /* Parent and child execute simultaneously */
    for (int i = 0; i < 3; i++) {
        plm_sleep (770);
        printf ("Parent execute\n");
    }

    time_to_die = true;

    return 0;
}
