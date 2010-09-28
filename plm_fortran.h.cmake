/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _plm_fortran_h_
#define _plm_fortran_h_

#cmakedefine PLM_USE_F2C
#cmakedefine PLM_USE_INCLUDED_F2C

#if defined (_WIN32)
  #include "@CMAKE_SOURCE_DIR@/plm_f2c_win32.h"

#else /* UNIX */

  #if defined (PLM_USE_F2C)
    #if defined (PLM_USE_INCLUDED_F2C)
       /* Included f2c. */
       #include "@CMAKE_SOURCE_DIR@/libs/libf2c/f2c.h"
    #else 
       /* System or self-built f2c.
          If you try to compile and link against a self-built f2c library 
          without installing, you can't include the f2c directory because 
          f2c includes a broken "ctype.h" which conflicts with the system one. 
	  Therefore, we use a full path here.
	  */
      #include "@F2C_INCLUDE_DIR@/f2c.h"
    #endif
  #else
    /* We're using a real fortran compiler.  Try the included f2c.h */
    #include "@CMAKE_SOURCE_DIR@/libs/libf2c/f2c.h"
  #endif
#endif /* UNIX */

#endif
