/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _ss_list_io_h_
#define _ss_list_io_h_

#include "plmbase_config.h"

class Rtss_structure_set;

PLMBASE_C_API Rtss_structure_set* ss_list_load (
        Rtss_structure_set* cxt,
        const char* ss_list_fn
);
PLMBASE_C_API void ss_list_save (
        Rtss_structure_set* cxt,
        const char* cxt_fn
);
PLMBASE_C_API void ss_list_save_colormap (
        Rtss_structure_set* cxt,
        const char* colormap_fn
);

#endif
