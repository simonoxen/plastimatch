/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _plm_path_h_
#define _plm_path_h_

/**
*  You probably do not want to #include this header directly.
 *
 *   Instead, it is preferred to #include "plmsys.h"
 */

#include <stdlib.h>
#include <limits.h>

/* Need a sane way of dealing with path buffers, see e.g.:
   http://insanecoding.blogspot.com/2007/11/pathmax-simply-isnt.html */
/* Here is what posix has to say on the subject:
   http://www.opengroup.org/onlinepubs/009695399/basedefs/limits.h.html */

#ifndef _MAX_PATH
#ifdef PATH_MAX
#define _MAX_PATH PATH_MAX
#else
#define _MAX_PATH 255
#endif
#endif

#endif
