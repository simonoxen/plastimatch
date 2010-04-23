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
    float *warped_landmarks; //moving landmarks displaced by current vector field
	int *landvox_mov;
	int *landvox_fix;
	int *landvox_warp;
	float *rbf_coeff;
	float *landmark_dxyz; //temporary array used in RBF
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

void bspline_landmarks_warp (
	Volume *vector_field, 
	BSPLINE_Parms *parms,
	BSPLINE_Xform* bxf, 
    Volume *fixed, 
    Volume *moving);

void bspline_landmarks_write_file( char *fn, char *title, float *coords, int n, float *offset);

#if defined __cplusplus
}
#endif

#endif
