/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _delayload_h_
#define _delayload_h_

// Needed for delay loading windows DLLs
#if _WIN32
    #pragma comment(lib, "delayimp")
    #pragma comment(lib, "user32")
#endif


// Note: if you attempt to load a library that
//       does not exist or cannot be found, this
//       returns a null pointer.
#define LOAD_LIBRARY(lib)                      \
    void* lib = dlopen (#lib".so", RTLD_LAZY);  

// Note: if lib contains a null pointer here
//       (see above note), this will return a
//       null function pointer.  Be careful pls.
#define LOAD_SYMBOL(sym, lib)                  \
    void (*sym)() = dlsym (lib, #sym);          

// Note: if lib contains a null pointer here
//       (see above note), this will return a
//       null function pointer.  Be careful pls.
#define LOAD_SYMBOL_SPECIAL(sym, lib, type)    \
    type (*sym)() = dlsym (lib, #sym);          

// Note: if lib does not point to a valid open
//       resource, then this will simply return
//       a non-zero value.  No biggie.
#define UNLOAD_LIBRARY(lib)                    \
    dlclose (lib);                              

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

#if defined __cplusplus
}
#endif

#endif
