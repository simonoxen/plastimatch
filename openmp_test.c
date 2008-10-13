#include "plm_config.h"
#include <stdio.h>
#if (OPENMP_FOUND)
#include <omp.h>
#endif

#if (OPENMP_FOUND)
void
display_num_threads (void)
{
    int nthreads, tid;

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
}

void
speedtest_openmp (void)
{
#define LOOP1 100000
#define LOOP2 100000
    int i, j;
    double d1, d2;
    d1 = 0.0;
    d2 = 1.0;
    for (i = 0; i < LOOP1; i++) {
	for (j = 0; j < LOOP2; j++) {
	    d1 += 0.872013;
	    d2 *= d1;
	    if (d2 > 1.0) {
		d2 -= floor(d2);
	    }
	}
    }
}
#endif /* OPENMP_FOUND */

int
main (int argc, char* argv)
{

#if (OPENMP_FOUND)
    display_num_threads ();
#else
    printf ("Sorry, openmp was not supported by your compiler\n");
#endif

    return 0;
}
