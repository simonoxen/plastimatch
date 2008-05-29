/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef __DEBUG_H__
#define __DEBUG_H__

#include "time.h"
#include <stdio.h>

#ifdef _WIN32
/* from <sys/cdefs.h> */
# ifdef  __cplusplus
#  define __BEGIN_DECLS  extern "C" {
#  define __END_DECLS    }
# else
#  define __BEGIN_DECLS
#  define __END_DECLS
# endif
# define __P(args)      args
#endif

__BEGIN_DECLS
void debug_open (char* fn);
void debug_close (void);
void debug_printf (char* fmt, ...);
void debug_enable (void);
__END_DECLS

#endif//__DEBUG_H__
