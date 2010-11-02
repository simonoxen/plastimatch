/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include <stdio.h>
#if defined (_WIN32)
#include <windows.h>	// Only needed for LoadLibrary()
#endif

/* Note:
 * Any utility that plans to use GPU functions should
 * call this function to check if the CUDA or OpenCL runtimes
 * are available.
 * 
 * Return 1 if runtime found, 0 if runtime not found.
 */

int 
delayload_cuda (void)
{
#if defined (_WIN32)
    if (LoadLibrary ("nvcuda.dll") == NULL) {
        printf ("Failed to load CUDA runtime!\n");
        printf ("Please install the CUDA runtime to enable GPU acceleration.\n");
        return 0;
    } else {
        return 1;
    }
#else
    // Assume linux users are compiling from source
    // and won't attempt to run features they don't
    // or can't utilize.
    return 1;
#endif
}

int 
delayload_opencl (void)
{
#if defined (_WIN32)
    if (LoadLibrary ("opencl.dll") == NULL) {
        printf ("opencl.dll not found!  Please install an OpenCL runtime.\n");
        return 0;
    } else {
        return 1;
    }
#else
    // Assume linux users are compiling from source
    // and won't attempt to run features they don't
    // or can't utilize.
    return 1;
#endif
}
