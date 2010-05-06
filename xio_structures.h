/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _xio_structures_h_
#define _xio_structures_h_

#include "plm_config.h"
#include "cxt_io.h"
#include "xio_io.h"

plastimatch1_EXPORT void
xio_structures_load (Cxt_structure_list *structures, char *input_dir, 
		     float x_adj, float y_adj, Xio_patient_position pt_position);
plastimatch1_EXPORT 
void
xio_structures_save (
    Cxt_structure_list *cxt, 
    Xio_version xio_version, 
    char *output_dir
);

#endif
