/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _bspline_landmarks_h_
#define _bspline_landmarks_h_

#include "plm_config.h"

typedef struct bspline_landmarks Bspline_landmarks;
struct bspline_landmarks {
    int num_landmarks;
    float *fixed_landmarks;
    float *moving_landmarks;
    int *landvox_mov;
};

#if defined __cplusplus
extern "C" {
#endif

Bspline_landmarks*
bspline_landmarks_load (char *fixed_fn, char *moving_fn);

void
bspline_landmarks_adjust (Bspline_landmarks *blm, Volume *fixed, Volume *moving);

void
bspline_landmarks_score (
    BSPLINE_Parms *parms, 
    Bspline_state *bst, 
    BSPLINE_Xform *bxf, 
    Volume *fixed, 
    Volume *moving
);

#if defined __cplusplus
}
#endif

#endif
