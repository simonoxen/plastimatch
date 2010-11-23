/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plm_config.h"
#include <stdio.h>
#if defined (_WIN32)
#include <windows.h>	// for LoadLibrary()
#else
#include <dlfcn.h>      // for dlopen()
#endif

/* Note:
 * Any utility that plans to use GPU functions should
 * call this function to check if the CUDA or OpenCL runtimes
 * are available.
 * 
 * Return 1 if runtime found, 0 if runtime not found.
 */

// Note: We need special cases for Windows and POSIX compliant OSes
int 
delayload_cuda (void)
{
#if defined (_WIN32)
    // Windows
    // For Windows we try to load the CUDA drivers:
    // * If they don't exist -> we should exit upon returning from this function.
    // * If they do exist    -> windows has been told to delay load plmcuda.dll
    //                          when a cuda function is encountered, we will
    //                          load plmcuda.dll automagically, which we know
    //                          will work as intended since we have 1st checked
    //                          that the cuda drivers are installed on the
    //                          users system (nvcuda.dll).  (See also
    //                          CMakeLists.txt for /DELAYLOAD configuration)

    // Because good ol' Windows can't do symlinks, there is no version safe
    // way to check for the runtime... the name of the file changes with every
    // new version release of the CUDA Toolkit.  Users will just need to read
    // the documentation and install the version of the toolkit that was used
    // to build the plastimatch CUDA plugin (plmcuda.dll) OR compile from
    // source.
    if ( (LoadLibrary ("nvcuda.dll") == NULL)      /* CUDA Driver */
#if defined (PLM_USE_CUDA_PLUGIN)
        || (LoadLibrary ("plmcuda.dll") == NULL)   /* PLM CUDA Plugin */
#endif
       ) {
        printf ("Failed to load CUDA runtime!\n");
        printf ("For GPU acceleration, please install:\n");
        printf ("* the plastimatch GPU plugin\n");
        printf ("* the CUDA Toolkit version needed by the GPU plugin\n\n");
        printf ("Visit http://www.plastimatch.org/contents.html for more information.\n\n");
        return 0;
    } else {
        // success
        return 1;
    }
#else
    // NOT Windows (most likely a POSIX compliant OS though...)
    //
    // Check for *both* the CUDA runtime & CUDA driver
    //
    // I think this is version safe due to the way nvidia does symlinks.
    // For example, on my system:
    //
    // libcuda.so -> libcuda.so.195.36.24
    // *and*
    // libcudart.so -> libcudart.so.3
    // libcudart.so.3 -> libcudart.so.3.0.14
    //
#if defined (PLM_USE_CUDA_PLUGIN)
    if ( (dlopen ("libcuda.so", RTLD_LAZY) == NULL)    ||   /* CUDA Driver */
         (dlopen ("libcudart.so", RTLD_LAZY) == NULL)  ||   /* CUDA RunTime */
         (dlopen ("libplmcuda.so", RTLD_LAZY) == NULL)      /* PLM CUDA Plugin */
       ) {
        printf ("Failed to load CUDA runtime!\n");
        printf ("For GPU acceleration, please install:\n");
        printf ("* the plastimatch GPU plugin\n");
        printf ("* the CUDA Toolkit version needed by the GPU plugin\n\n");
        printf ("Visit http://www.plastimatch.org/contents.html for more information.\n\n");
        return 0;
    } else {
        // success
        return 1;
    }
#else
    return 1;
#endif

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
