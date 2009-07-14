/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _plm_f2c_h_
#define _plm_f2c_h_

#if defined (_WIN32)
#include "plm_f2c_win32.h"
#else
/* If you try to compile and link against a self-built f2c library 
   without installing, you can't include the f2c directory because 
   f2c includes a broken "ctype.h" which conflicts with the system one.
*/
#include "@F2C_INCLUDE_DIR@/f2c.h"
#endif

#endif
