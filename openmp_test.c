#include "plm_config.h"
#if (OPENMP_FOUND)
#include <omp.h>
#endif

void
speedtest_ref (void)
{
    int nthreads, tid;
#if (OPENMP_FOUND)

    /* Run a simple speed test, to see how fast things run */
#pragma omp parallel private(tid)
    {

	/* Obtain and print thread id */
	tid = omp_get_thread_num();
	printf("Hello World from thread = %d\n", tid);

	/* Only master thread does this */
	if (tid == 0) 
	{
	    nthreads = omp_get_num_threads();
	    printf("Number of threads = %d\n", nthreads);
	}

    }  /* All threads join master thread and terminate */
#endif
}

void
speedtest_openmp (void)
{
}

int
main (int argc, char* argv)
{
    int nthreads, tid;
#if (OPENMP_FOUND)
    /* Fork a team of threads giving them their own copies of variables */
#pragma omp parallel private(tid)
    {

	/* Obtain and print thread id */
	tid = omp_get_thread_num();
	printf("Hello World from thread = %d\n", tid);

	/* Only master thread does this */
	if (tid == 0) 
	{
	    nthreads = omp_get_num_threads();
	    printf("Number of threads = %d\n", nthreads);
	}

    }  /* All threads join master thread and terminate */
#endif

    return 0;
}

