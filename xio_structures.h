/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _xio_structures_h_
#define _xio_structures_h_

#include "plm_config.h"
#include "cxt_io.h"
#include "xio_ct.h"

plastimatch1_EXPORT void
xio_structures_load (Cxt_structure_list *structures, char *input_dir);
plastimatch1_EXPORT 
void
xio_structures_save (
    Cxt_structure_list *cxt, 
    Xio_ct_transform *transform, 
    Xio_version xio_version, 
    char *output_dir
);
plastimatch1_EXPORT void
xio_structures_apply_transform (Cxt_structure_list *structures, Xio_ct_transform *transform);

#endif
