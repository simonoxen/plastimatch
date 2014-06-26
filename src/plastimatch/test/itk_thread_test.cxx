#include <stdio.h>
#include "itkConditionVariable.h"
#include "itkMultiThreader.h"
#include "itkMutexLock.h"
#include "itkMutexLockHolder.h"
#include "plm_sleep.h"

itk::SimpleMutexLock *sfml;

class Thread_struct
{
public:
    itk::SimpleMutexLock mutex;
    itk::ConditionVariable::Pointer condition;
    bool semaphore_available;
    int j;
    bool die;
public:
    Thread_struct () {
        condition = itk::ConditionVariable::New(); 
        die = false;
        j = 30;
        semaphore_available = true;
        mutex.Unlock ();
    }
    ~Thread_struct () {
    }
    void release_semaphore ()
    {
        mutex.Lock();
        semaphore_available = true;
        condition->Signal();
#if defined (commentout)
#endif
        mutex.Unlock();
    }
    void grab_semaphore ()
    {
        mutex.Lock();
        if (semaphore_available == false) {
            condition->Wait (&mutex);
        }
        semaphore_available = false;
        mutex.Unlock();
#if defined (commentout)
#endif
    }
};


ITK_THREAD_RETURN_TYPE 
thread_func (void* param)
{
    itk::MultiThreader::ThreadInfoStruct *info 
        = (itk::MultiThreader::ThreadInfoStruct*) param;
    Thread_struct* ts = (Thread_struct*) info->UserData;
    while (1) {
        //printf ("Child: %d\n", ts->semaphore_available);
        ts->grab_semaphore ();
        plm_sleep (300);
        printf ("Child execute\n");
        ++ ts->j;
        ts->release_semaphore ();
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
    printf ("Parent: %d\n", ts.semaphore_available);
    ts.grab_semaphore ();
    printf ("Parent only.\n");
    for (int i = 0; i < 15; i++) {
        plm_sleep (70);
        printf ("Parent.\n");
    }
    printf ("End parent only.\n");
    ts.release_semaphore ();
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
