/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _bspline_h_
#define _bspline_h_

#include "plmregister_config.h"
#include "plm_int.h"
#include <string>

class Bspline_mi_hist_set;
class Bspline_optimize;
class Bspline_parms;
class Bspline_score;
class Bspline_state;
class Bspline_xform;
class Volume;

PLMREGISTER_API Volume* bspline_compute_vf (const Bspline_xform* bxf);
void bspline_display_coeff_stats (Bspline_xform* bxf);
PLMREGISTER_API void bspline_score (Bspline_optimize *bod);
void bspline_update_smetric_grad (
    Bspline_state *bst, 
    Bspline_xform* bxf, 
    plm_long p[3], plm_long qidx, float dc_dv[3]
);
void bspline_update_smetric_grad_b (
    Bspline_score* bscore,
    const Bspline_xform* bxf, 
    plm_long pidx, 
    plm_long qidx, 
    const float dc_dv[3]
);
int* calc_offsets (int* tile_dims, int* cdims);
void find_knots (plm_long* knots, plm_long tile_num, plm_long* rdims, plm_long* cdims);
void report_score (
    Bspline_parms *parms,
    Bspline_xform *bxf, 
    Bspline_state *bst
);

/* Debugging routines */
PLMREGISTER_API void bspline_save_debug_state (
    Bspline_parms *parms, 
    Bspline_state *bst, 
    Bspline_xform* bxf
);
void dump_xpm_hist (Bspline_mi_hist_set* mi_hist, char* file_base, int iter);
void bspline_condense_smetric_grad (
    float* cond_x, float* cond_y, float* cond_z,
    Bspline_xform* bxf,
    Bspline_score* ssd
);
void bspline_update_sets (
    float* sets_x, float* sets_y, float* sets_z,
    int qidx,
    float* dc_dv,
    Bspline_xform* bxf
);
void bspline_update_sets_b (
    float* sets_x, float* sets_y, float* sets_z,
    plm_long *q,
    float* dc_dv,
    Bspline_xform* bxf
);
void bspline_sort_sets (
    float* cond_x, float* cond_y, float* cond_z,
    float* sets_x, float* sets_y, float* sets_z,
    plm_long pidx,
    Bspline_xform* bxf
);

#endif
