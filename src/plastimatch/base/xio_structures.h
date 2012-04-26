/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _xio_structures_h_
#define _xio_structures_h_

#include "plmbase_config.h"
#include "cxt_io.h"
#include "metadata.h"
#include "xio_ct.h"
#include "xio_studyset.h"

plastimatch1_EXPORT void
xio_structures_load (Rtss_polyline_set *structures, const Xio_studyset& xsl);

plastimatch1_EXPORT 
void
xio_structures_save (
    Rtss_polyline_set *cxt, 
    Metadata *meta,
    Xio_ct_transform *transform, 
    Xio_version xio_version, 
    const char *output_dir
);

plastimatch1_EXPORT void
xio_structures_apply_transform (Rtss_polyline_set *structures, Xio_ct_transform *transform);

#endif
