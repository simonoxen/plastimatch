#include <stdio.h>
#include "itkMultiThreader.h"
#include "itkSemaphore.h"
#include "itkSimpleFastMutexLock.h"
#include "plm_sleep.h"

itk::SimpleFastMutexLock *sfml;

class Thread_struct
{
public:
    itk::Semaphore::Pointer semaphore;
    int j;
    bool die;
public:
    Thread_struct () {
        die = false;
        j = 30;
        semaphore = itk::Semaphore::New ();
        semaphore->Initialize (1);
    }
    ~Thread_struct () {
    }
};


ITK_THREAD_RETURN_TYPE 
thread_func (void* param)
{
    itk::MultiThreader::ThreadInfoStruct *info 
        = (itk::MultiThreader::ThreadInfoStruct*) param;
    Thread_struct* ts = (Thread_struct*) info->UserData;
    while (1) {
        ts->semaphore->Down ();
        plm_sleep (300);
        ++ ts->j;
        printf ("Child: %d\n", ts->j);
        ts->semaphore->Up ();
        if (ts->die) {
            break;
        }
    }
    return ITK_THREAD_RETURN_VALUE;
}

int main ()
{
    Thread_struct ts;

    itk::MultiThreader::Pointer threader = itk::MultiThreader::New();

    printf ("Gonna Spawn...\n");
    int thread_no = threader->SpawnThread (thread_func, (void*) &ts);
    for (int i = 0; i < 3; i++) {
        plm_sleep (770);
        printf ("Parent.\n");
    }
    ts.semaphore->Down();
    for (int i = 0; i < 8; i++) {
        plm_sleep (770);
        printf ("Parent.\n");
    }
    ts.semaphore->Up();
    for (int i = 0; i < 3; i++) {
        plm_sleep (770);
        printf ("Parent.\n");
    }

    ts.die = true;
    printf ("Gonna wait..\n");
    threader->TerminateThread (thread_no);

    printf ("final j = %d\n", ts.j);
    return 0;
}
