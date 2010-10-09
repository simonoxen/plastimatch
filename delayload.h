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

#if defined __cplusplus
extern "C" {
#endif

gpuit_EXPORT
void cudaDetect();

#if defined __cplusplus
}
#endif

#endif
