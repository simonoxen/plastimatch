/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _xio_structures_h
#define _xio_structures_h

#include "plm_config.h"
#include "cxt_io.h"

plastimatch1_EXPORT void
xio_structures_load (Cxt_structure_list *structures, char *input_dir, 
		     float x_adj, float y_adj);

#endif
