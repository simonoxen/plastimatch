/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef __dcmtk_config_h__
#define __dcmtk_config_h__

/* Debian OS install of DCMTK is broken.  This is the workaround. */
#if DCMTK_HAVE_CONFIG_H
#define HAVE_CONFIG_H 1
#endif

/* Make sure OS specific configuration is included before other
   dcmtk headers. */
#include "dcmtk/config/osconfig.h"

#endif /* __dcmtk_config_h__ */
