/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _plm_int_h_
#define _plm_int_h_

#include "plmsys_config.h"

#if defined (GDCMCOMMON_H)
/* Gdcm 1.x has a broken header file gdcmCommon.h, which defines C99 types 
   (e.g. int32_t) when missing on MSVC.  It conflicts with plm_int.h 
   (which also fixes missing C99 types).  Here is a workaround for 
   this issue. */
#if !defined (CMAKE_HAVE_STDINT_H) && !defined (CMAKE_HAVE_INTTYPES_H) \
    && (defined(_MSC_VER)                                              \
        || defined(__BORLANDC__) && (__BORLANDC__ < 0x0560)            \
        || defined(__MINGW32__))
typedef __int64 int64_t;
typedef unsigned __int64 uint64_t;
#endif /* GDCMCOMMON_H */
#elif defined (_MSC_VER) && (_MSC_VER < 1600)
#include "msinttypes/stdint.h"
#else
#include <stdint.h>
#endif

/* The data type plm_long is a signed integer with the same size as size_t.
   It is equivalent to the POSIX idea of ssize_t.  It is used for 
   OpenMP 2.0 loop variables which must be signed. */
#if (CMAKE_SIZEOF_SIZE_T == 8)
typedef int64_t plm_long;
#elif (CMAKE_SIZEOF_SIZE_T == 4)
typedef int32_t plm_long;
#else
#error "Unexpected value for sizeof(size_t)"
#endif

#endif