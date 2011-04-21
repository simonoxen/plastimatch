/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _print_and_exit_h_
#define _print_and_exit_h_

#include "plm_config.h"


#define error_printf(fmt, ...) \
    fprintf (stderr, "\nplastimatch has encountered an issue.\n" \
             "file: %s (line:%i)\n" fmt, __FILE__, __LINE__,##__VA_ARGS__)



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
