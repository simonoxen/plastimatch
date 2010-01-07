/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _cxt_to_mha_h_
#define _cxt_to_mha_h_

#include "plm_config.h"
#include "cxt_io.h"
#include "volume.h"

class Cxt_to_mha_state {
public:
    bool want_prefix_imgs;
    bool want_labelmap;
    bool want_ss_img;

    unsigned char* acc_img;
    Volume* uchar_vol;
    Volume* labelmap_vol;
    Volume* ss_img_vol;

    int curr_struct_no;
};

#if defined __cplusplus
extern "C" {
#endif

plastimatch1_EXPORT
void
cxt_to_mha_init (
    Cxt_to_mha_state *ctm_state,
    Cxt_structure_list *structures,
    bool want_prefix_imgs,
    bool want_labelmap,
    bool want_ss_img
);
plastimatch1_EXPORT
bool
cxt_to_mha_process_next (
    Cxt_to_mha_state *ctm_state,
    Cxt_structure_list *structures
);
plastimatch1_EXPORT
const char*
cxt_to_mha_current_name (
    Cxt_to_mha_state *ctm_state,
    Cxt_structure_list *structures
);
plastimatch1_EXPORT
void
cxt_to_mha_free (Cxt_to_mha_state *ctm_state);

#if defined __cplusplus
}
#endif

#endif
