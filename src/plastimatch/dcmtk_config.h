/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _dcmtk_config_h_
#define _dcmtk_config_h_

/* Debian OS install of DCMTK is broken.  This is the workaround. */
#include "plm_config.h"
#if DCMTK_HAVE_CFUNIX_H
#define HAVE_CONFIG_H 1
#endif

/* Make sure OS specific configuration is included before other
   dcmtk headers. */
#include "dcmtk/config/osconfig.h"

#endif /* __dcmtk_config_h__ */
