/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _plm_version_h_
#define _plm_version_h_

#include "plm_config.h"

#if (PLASTIMATCH_EXPERIMENTAL)
  #define PLASTIMATCH_VERSION "1.4-beta"
  #define PLASTIMATCH_VERSION_STRING  PLASTIMATCH_VERSION " (" PLASTIMATCH_BUILD_NUMBER ")"
#else
  #define PLASTIMATCH_VERSION "1.4.0"
  #define PLASTIMATCH_VERSION_STRING  PLASTIMATCH_VERSION
#endif

#endif
