/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _bspline_mi_h_
#define _bspline_mi_h_

#include "plm_config.h"
#include "bspline.h"
#include "volume.h"

/* -----------------------------------------------------------------------
   Function declarations
   ----------------------------------------------------------------------- */
#if defined __cplusplus
extern "C" {
#endif

gpuit_EXPORT
void
bspline_initialize_mi (Bspline_parms* parms);

gpuit_EXPORT
void
bspline_score_c_mi (
    Bspline_parms *parms,
    Bspline_state *bst, 
    Bspline_xform *bxf
);

gpuit_EXPORT
void
bspline_score_d_mi (
    Bspline_parms *parms,
    Bspline_state *bst, 
    Bspline_xform *bxf
);

gpuit_EXPORT
void
bspline_score_e_mi (
    Bspline_parms *parms,
    Bspline_state *bst, 
    Bspline_xform *bxf
);

gpuit_EXPORT
void
bspline_score_f_mi (
    Bspline_parms *parms,
    Bspline_state *bst, 
    Bspline_xform *bxf
);

gpuit_EXPORT
void
bspline_score_g_mi (
    Bspline_parms *parms, 
    Bspline_state *bst,
    Bspline_xform *bxf
);

gpuit_EXPORT
void
bspline_score_h_mi (
    Bspline_parms *parms, 
    Bspline_state *bst,
    Bspline_xform *bxf
);

#if defined __cplusplus
}
#endif

#endif
