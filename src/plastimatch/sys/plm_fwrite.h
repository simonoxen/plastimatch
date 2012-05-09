/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _plm_fwrite_h_
#define _plm_fwrite_h_

/**
*  You probably do not want to #include this header directly.
 *
 *   Instead, it is preferred to #include "plmsys.h"
 */

#include "plmsys_config.h"

PLMSYS_C_API void plm_fwrite (
        void* buf,
        size_t size,
        size_t count,
        FILE* fp, 
        bool force_little_endian
);

#endif

