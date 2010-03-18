#ifndef _bspline_old_h_
#define _bspline_old_h_





#if defined __cplusplus
extern "C" {
#endif

static void
bspline_interp_pix_b (float out[3], BSPLINE_Xform* bxf, int pidx, int qidx);


inline void
clip_and_interpolate_obsolete (
    Volume* moving,	/* Moving image */
    float* dxyz,	/* Vector displacement of current voxel */
    float* dxyzf,	/* Floor of vector displacement */
    int d,		/* 0 for x, 1 for y, 2 for z */
    int* maf,		/* x, y, or z coord of "floor" pixel in moving img */
    int* mar,		/* x, y, or z coord of "round" pixel in moving img */
    int a,		/* Index of base voxel (before adding displacemnt) */
    float* fa1,		/* Fraction of interpolant for lower index voxel */
    float* fa2		/* Fraction of interpolant for upper index voxel */
);

void bspline_score_f_mse (BSPLINE_Parms *parms,
			  Bspline_state *bst, 
			  BSPLINE_Xform *bxf,
			  Volume *fixed,
			  Volume *moving,
			  Volume *moving_grad);


void
bspline_score_e_mse (BSPLINE_Parms *parms, 
		     Bspline_state *bst,
		     BSPLINE_Xform* bxf, 
		     Volume *fixed, 
		     Volume *moving, 
		     Volume *moving_grad);


void
bspline_score_d_mse (BSPLINE_Parms *parms, 
		     Bspline_state *bst,
		     BSPLINE_Xform* bxf, 
		     Volume *fixed, 
		     Volume *moving, 
		     Volume *moving_grad);

void
bspline_score_c_mse (
    BSPLINE_Parms *parms, 
    Bspline_state *bst,
    BSPLINE_Xform* bxf, 
    Volume *fixed, 
    Volume *moving, 
    Volume *moving_grad
);

void
bspline_score_b_mse 
(
 BSPLINE_Parms *parms, 
 Bspline_state *bst,
 BSPLINE_Xform *bxf, 
 Volume *fixed, 
 Volume *moving, 
 Volume *moving_grad);

void
bspline_score_a_mse 
(
 BSPLINE_Parms *parms, 
 Bspline_state *bst,
 BSPLINE_Xform* bxf, 
 Volume *fixed, 
 Volume *moving, 
 Volume *moving_grad
 );

#if defined __cplusplus
}
#endif


#endif
