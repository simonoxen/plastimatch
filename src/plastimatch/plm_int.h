/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _plm_int_h_
#define _plm_int_h_

#include "plm_config.h"

#if defined (_MSC_VER) && (_MSC_VER < 1600)
#include "msinttypes/stdint.h"
#else
#include <stdint.h>
#endif

#if (CMAKE_SIZEOF_SIZE_T == 8)
typedef int64_t plm_long;
#elif (CMAKE_SIZEOF_SIZE_T == 4)
typedef int32_t plm_long;
#else
#error "Unexpected value for sizeof(size_t)"
#endif

#endif
