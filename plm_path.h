/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _plm_path_h_
#define _plm_path_h_

#include <stdio.h>

/* Need a sane way of dealing with path buffers.
   http://insanecoding.blogspot.com/2007/11/pathmax-simply-isnt.html */
#ifndef _MAX_PATH
#ifdef PATH_MAX
#define _MAX_PATH PATH_MAX
#else
#define _MAX_PATH 256
#endif
#endif

#endif
