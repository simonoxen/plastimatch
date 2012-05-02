/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _mha_io_h_
#define _mha_io_h_

/**
*  You probably do not want to #include this header directly.
 *
 *   Instead, it is preferred to #include "plmbase.h"
 */

#include "plmbase_config.h"

class Volume;

C_API Volume* read_mha (const char* filename);
C_API void write_mha (const char* filename, Volume* vol);

#endif
