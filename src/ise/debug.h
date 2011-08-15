/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef __DEBUG_H__
#define __DEBUG_H__

#include <stdio.h>

#if defined __cplusplus
extern "C" {
#endif

void debug_open (void);
void debug_close (void);
void debug_printf (char* fmt, ...);
void debug_enable (void);

void mprintf (char* fmt, ...);

#if defined __cplusplus
}
#endif

#endif
