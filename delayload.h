/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _delayload_h_
#define _delayload_h_

#include "plm_config.h"
#ifndef _WIN32
#include <dlfcn.h>
#endif

#if (OPENCL_FOUND)
#include "delayload_opencl.h"
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
        void (*sym)() = dlsym (lib, #sym);          
#else
    #define LOAD_SYMBOL(sym, lib)                  \
        ;
#endif

// Note: if lib contains a null pointer here
//       (see above note), this will return a
//       null function pointer.  Be careful pls.
#if !defined(_WIN32) && defined(PLM_USE_GPU_PLUGINS)
    #define LOAD_SYMBOL_SPECIAL(sym, lib, type)    \
        type (*sym)() = dlsym (lib, #sym);   
#else
    #define LOAD_SYMBOL_SPECIAL(sym, lib, type)    \
        ;
#endif


// JAS 2010.12.10
// OpenCL is special since the OpenCL dev stuff
// doesn't include return typedefs... so we make them
// oursevles (see delayload_opencl.h) and we have a
// special casting dlsym() macro for it.
#if !defined(_WIN32) && defined(PLM_USE_GPU_PLUGINS)
    #define LOAD_SYMBOL_OPENCL(sym, lib, type)    \
        sym = (__##sym *)dlsym (lib, #sym);   
#else
    #define LOAD_SYMBOL_SPECIAL(sym, lib, type)    \
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


// Note: Windows will not automatically export
//       symbols for external linking, so we have
//       to tell it which symbols (DLL interface
//       functions) to export.  gcc does not need
//       this.  Without this, link.exe will not
//       generate plmcuda.lib, which is needed to
//       use plmcuda.dll
#if _WIN32
    #define plmcuda_EXPORT  \
    __declspec(dllexport)    
#else
    #define plmcuda_EXPORT ;
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
