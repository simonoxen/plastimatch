/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plm_config.h"
#include <stdio.h>
#include <time.h>
#if (OPENMP_FOUND)
#include <omp.h>
#endif

#define LOOP1 2000
#define LOOP2 20000

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
speedtest_openmp_1 (void)
{
    int i;
#pragma omp parallel for
    for (i = 0; i < LOOP1; i++) {
	int j;
	double d1 = 0.0;
	double d2 = 1.0;
	for (j = 0; j < LOOP2; j++) {
	    d1 += 0.872013;
	    d2 *= d1;
	    if (d2 > 1.0) {
		d2 -= floor(d2);
	    }
	}
    }
}

void
speedtest_openmp_2 (void)
{
    int i;
    double d1, d2;
    for (i = 0; i < LOOP1; i++) {
	int j;
	double d1 = 0.0;
	double d2 = 1.0;
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
    clock_t time_start, time_end;

#if (OPENMP_FOUND)
    display_num_threads ();
    time_start = clock ();
    speedtest_openmp_1 ();
    time_end = clock ();
    printf ("Time = %f seconds\n", 
	    ((float) (time_end - time_start)) / CLOCKS_PER_SEC);

    time_start = clock ();
    speedtest_openmp_2 ();
    time_end = clock ();
    printf ("Time = %f seconds\n", 
	    ((float) (time_end - time_start)) / CLOCKS_PER_SEC);

#else
    printf ("Sorry, openmp was not supported by your compiler\n");
#endif

    return 0;
}
