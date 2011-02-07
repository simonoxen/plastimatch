/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _delayload_h_
#define _delayload_h_

#include "plm_config.h"
#ifndef _WIN32
#include <dlfcn.h>
#endif

// Needed for delay loading windows DLLs
#if _WIN32
    #pragma comment(lib, "delayimp")
    #pragma comment(lib, "user32")
#endif

// JAS 2010.11.23
// ------------------------------------------------------------
// Because these macros contain declarations, their
// usage is restricted under C89.  C89-GNU, C99, and
// C++ are all cool with using these macros pretty
// much anywhere... be careful inside switch statements.
// Still, their usage will determine the portability of
// plastimatch.

// Note: if you attempt to load a library that
//       does not exist or cannot be found, this
//       returns a null pointer.
#if !defined(_WIN32) && defined(PLM_USE_GPU_PLUGINS)
    #define LOAD_LIBRARY(lib)                      \
        void* lib = dlopen_ex (#lib".so");          
#else
    #define LOAD_LIBRARY(lib)                      \
        ;
#endif

// Note: if lib contains a null pointer here
//       (see above note), this will return a
//       null function pointer.  Be careful pls.
#if !defined(_WIN32) && defined(PLM_USE_GPU_PLUGINS)
    #define LOAD_SYMBOL(sym, lib)                  \
        sym##_##t* sym = (sym##_##t*) dlsym (lib, #sym);          
#else
    #define LOAD_SYMBOL(sym, lib)                  \
        ;
#endif

// ------------------------------------------------------------

// JAS 2010.12.09
// Despite what the man pages say, dlclose()ing NULL
// was resulting in segfaults!  So, now we check 1st.
#if !defined(_WIN32) && defined(PLM_USE_GPU_PLUGINS)
    #define UNLOAD_LIBRARY(lib)                    \
        if (lib != NULL) {                         \
            dlclose (lib);                         \
        }                                           
#else
    #define UNLOAD_LIBRARY(lib)                    \
        ;
#endif


#if defined __cplusplus
extern "C" {
#endif

gpuit_EXPORT
int
delayload_cuda (void);

gpuit_EXPORT
int
delayload_opencl (void);

gpuit_EXPORT
void* dlopen_ex (char* lib);

#if defined __cplusplus
}
#endif

#endif
