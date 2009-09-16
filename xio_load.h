/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _xio_load_h
#define _xio_load_h

#include "plm_config.h"
#include "cxt_io.h"

void
xio_load_structures (Cxt_structure_list *structures, char *input_dir, 
		     float x_adj, float y_adj);

#endif
