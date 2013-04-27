/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _xio_structures_h_
#define _xio_structures_h_

#include "plmbase_config.h"

#include "xio_studyset.h"

class Rtss;
class Metadata;
class Xio_studyset;

PLMBASE_C_API void xio_structures_load (
        Rtss *structures,
        const Xio_studyset& xsl
);

PLMBASE_C_API void xio_structures_save (
        Rtss *cxt, 
        Metadata *meta,
        Xio_ct_transform *transform, 
        Xio_version xio_version, 
        const char *output_dir
);

PLMBASE_C_API void xio_structures_apply_transform (
        Rtss *structures,
        Xio_ct_transform *transform
);

#endif
