/* -------------------------------------------------------------------------*
    See COPYRIGHT for copyright information.
 * -------------------------------------------------------------------------*/
#ifndef __DEBUG_H__
#define __DEBUG_H__

#include <stdio.h>

void debug_open (void);
void debug_close (void);
void debug_printf (char* fmt, ...);
void debug_enable (void);

void mprintf (char* fmt, ...);

#endif//__DEBUG_H__
