/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _print_and_exit_h_
#define _print_and_exit_h_

#include "plm_config.h"

#if defined __cplusplus
extern "C" {
#endif
plmsys_EXPORT 
void print_and_wait (char* prompt_fmt, ...);
plmsys_EXPORT 
void print_and_exit (char* prompt_fmt, ...);
#if defined __cplusplus
}
#endif

#endif
