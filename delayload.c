/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include <stdio.h>
#if defined (_WIN32)
#include <windows.h>	// Only needed for LoadLibrary()
#endif


/* Note:
 * Any utility that plans to use GPU functions should
 * call this function 1st to make sure the CUDA runtime
 * is available.
 */
void
cudaDetect()
{
#if defined (_WIN32)
	if (LoadLibrary ("nvcuda.dll") == NULL) {
		// Failure: CUDA runtime not available
        printf ("cudaDetect says, \"nvcuda.dll NOT found!\"\n");
        exit (0);
	} else {
        // do nothing...
	}
#else
    // Assume linux users are compiling from source
    // and won't attempt to run features they don't
    // or can't utilize.
#endif
}
