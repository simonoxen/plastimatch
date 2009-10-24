/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _bspline_h_
#define _bspline_h_

#include "volume.h"

enum BsplineOptimization {
    BOPT_LBFGSB,
    BOPT_STEEPEST
};

enum BsplineThreading {
    BTHR_CPU,
    BTHR_BROOK,
    BTHR_CUDA
};

enum BsplineMetric {
    BMET_MSE,
    BMET_MI
};

typedef struct BSPLINE_Xform_struct BSPLINE_Xform;
struct BSPLINE_Xform_struct {
    float img_origin[3];         /* Image origin (in mm) */
    float img_spacing[3];        /* Image spacing (in mm) */
    int img_dim[3];              /* Image size (in vox) */
    int roi_offset[3];	         /* Position of first vox in ROI (in vox) */
    int roi_dim[3];				 /* Dimension of ROI (in vox) */
    int vox_per_rgn[3];	         /* Knot spacing (in vox) */
    float grid_spac[3];          /* Knot spacing (in mm) */
    int rdims[3];                /* # of regions in (x,y,z) */
    int cdims[3];                /* # of knots in (x,y,z) */
    int num_knots;               /* Total number of knots (= product(cdims)) */
    int num_coeff;               /* Total number of coefficents (= product(cdims) * 3) */
    float* coeff;                /* Coefficients.  Vector directions interleaved. */
    int* cidx_lut;               /* Lookup volume for region number */
    int* c_lut;                  /* Lookup table for control point indices */
    int* qidx_lut;               /* Lookup volume for region offset */
    float* q_lut;                /* Lookup table for influence multipliers */
};

typedef struct BSPLINE_Score_struct BSPLINE_Score;
struct BSPLINE_Score_struct {
    float score;
    float* grad;
};


typedef struct dev_pointers_bspline Dev_Pointers_Bspline;
struct dev_pointers_bspline
{

	// IMPORTANT!
	// Each member of this struct is a POINTER TO
	// AN ADDRESS RESIDING IN THE GPU'S GLOBAL
	// MEMORY!  Care must be taken when referencing
	// and dereferencing members of this structure!

	float* my_gpu_addr;		// Holds address of this
					//   structure in global
					//   device memory.
	float* fixed_image;		// Fixed Image Voxels
	float* moving_image;		// Moving Image Voxels
	float* moving_grad;		// dc_dp (Gradient) Volume
	float* coeff;			// B-Spline coefficients (p)
	float* score;			// The "Score"
	float* dc_dv;			// dc_dv (Interleaved)
	float* dc_dv_x;			// dc_dv (De-Interleaved)
	float* dc_dv_y;			// dc_dv (De-Interleaved)
	float* dc_dv_z;			// dc_dv (De-Interleaved)
	float* cond_x;			// dc_dv_x (Condensed)
	float* cond_y;			// dc_dv_y (Condensed)
	float* cond_z;			// dc_dv_z (Condensed)
	float* grad;			// dc_dp
	float* dc_dp_x;
	float* dc_dp_y;
	float* dc_dp_z;
	float* grad_temp;
	int* LUT_Knot;
	int* LUT_NumTiles;
	int* LUT_Offsets;
	float* LUT_Bspline_x;
	float* LUT_Bspline_y;
	float* LUT_Bspline_z;

	// These hold the size of the
	// chucks of memory we allocated
	// that each start at the addresses
	// stored in the pointers above.
	size_t my_size;
	size_t fixed_image_size;
	size_t moving_image_size;
	size_t moving_grad_size;
	size_t coeff_size;
	size_t score_size;
	size_t dc_dv_size;
	size_t dc_dv_x_size;
	size_t dc_dv_y_size;
	size_t dc_dv_z_size;
	size_t cond_x_size;
	size_t cond_y_size;
	size_t cond_z_size;
	size_t grad_size;
	size_t grad_temp_size;
	size_t LUT_Knot_size;
	size_t LUT_NumTiles_size;
	size_t LUT_Offsets_size;
	size_t LUT_Bspline_x_size;
	size_t LUT_Bspline_y_size;
	size_t LUT_Bspline_z_size;
};


typedef struct bspline_state Bspline_state;
struct bspline_state {
    int it;
    BSPLINE_Score ssd;                   /* Score and Gradient */
    Dev_Pointers_Bspline* dev_ptrs;      /* GPU Device Pointers */
};

typedef struct BSPLINE_MI_Hist_Parms_struct BSPLINE_MI_Hist_Parms;
struct BSPLINE_MI_Hist_Parms_struct {
    long bins;
    float offset;
    float delta;
};

typedef struct BSPLINE_MI_Hist_struct BSPLINE_MI_Hist;
struct BSPLINE_MI_Hist_struct {
    BSPLINE_MI_Hist_Parms moving;
    BSPLINE_MI_Hist_Parms fixed;
    float* m_hist;
    float* f_hist;
    float* j_hist;
};

typedef struct BSPLINE_Parms_struct BSPLINE_Parms;
struct BSPLINE_Parms_struct {
    enum BsplineThreading threading;
    enum BsplineOptimization optimization;
    enum BsplineMetric metric;
    int max_its;
    int debug;                           /* Create grad & histogram files */
    char implementation;                 /* Which implementation ('a', 'b', etc.) */
    double convergence_tol;              /* When to stop iterations based on score */
    int convergence_tol_its;             /* How many iterations to check for convergence tol */
    BSPLINE_MI_Hist mi_hist;             /* Histogram for MI score */
    void *data_on_gpu;                   /* Pointer to structure encapsulating the data stored on the GPU */
    void *data_from_gpu;                 /* Pointer to structure that stores the data returned from the GPU */
};

 


#if defined __cplusplus
extern "C" {
#endif
gpuit_EXPORT
void bspline_parms_set_default (BSPLINE_Parms* parms);
gpuit_EXPORT
void bspline_xform_set_default (BSPLINE_Xform* bxf);
gpuit_EXPORT
Bspline_state *
bspline_state_create (BSPLINE_Xform *bxf);
gpuit_EXPORT
void
bspline_xform_initialize (
	BSPLINE_Xform* bxf,	         /* Output: bxf is initialized */
	float img_origin[3],         /* Image origin (in mm) */
	float img_spacing[3],        /* Image spacing (in mm) */
	int img_dim[3],              /* Image size (in vox) */
	int roi_offset[3],	         /* Position of first vox in ROI (in vox) */
	int roi_dim[3],		         /* Dimension of ROI (in vox) */
	int vox_per_rgn[3]);	     /* Knot spacing (in vox) */
gpuit_EXPORT
void bspline_xform_free (BSPLINE_Xform* bxf);
gpuit_EXPORT
void bspline_parms_free (BSPLINE_Parms* parms);
gpuit_EXPORT
void
bspline_state_free (Bspline_state *bst);
gpuit_EXPORT
void
bspline_warp (
    Volume *vout,         /* Output image (already sized and allocated) */
    Volume *vf_out,       /* Output vf (already sized and allocated, can be null) */
    BSPLINE_Xform* bxf,   /* Bspline transform coefficients */
    Volume *moving,       /* Input image */
    float default_val     /* Fill in this value outside of image */
);
gpuit_EXPORT
void bspline_optimize (BSPLINE_Xform* bxf, 
		       Bspline_state **bst,
		       BSPLINE_Parms *parms, 
		       Volume *fixed, 
		       Volume *moving, 
		       Volume *moving_grad);
gpuit_EXPORT
BSPLINE_Xform* read_bxf (char* filename);
gpuit_EXPORT
void write_bxf (char* filename, BSPLINE_Xform* bxf);
gpuit_EXPORT
void
bspline_interpolate_vf (Volume* interp, 
			BSPLINE_Xform* bxf);

/* Used internally */
void
bspline_set_coefficients (BSPLINE_Xform* bxf, float val);
void
bspline_display_coeff_stats (BSPLINE_Xform* bxf);
void
bspline_score (BSPLINE_Parms *parms, 
	       Bspline_state *bst,
	       BSPLINE_Xform* bxf, 
	       Volume *fixed, 
	       Volume *moving, 
	       Volume *moving_grad);
void
bspline_score_reference (BSPLINE_Score *ssd, 
			 Volume *fixed, Volume *moving, Volume *moving_grad, 
			 BSPLINE_Parms *parms);
void
clamp_linear_interpolate (float ma, int dmax, int* maf, int* mar, 
			  float* fa1, float* fa2);

void
bspline_update_grad_b (Bspline_state* bst, BSPLINE_Xform* bxf, 
		       int pidx, int qidx, float dc_dv[3]);
int* calc_offsets (int* tile_dims, int* cdims);

void find_knots (int* knots, int tile_num, int* rdims, int* cdims);
void
report_score (char *alg, BSPLINE_Xform *bxf, 
	      Bspline_state *bst, int num_vox, double timing);

/* Debugging routines */
void
dump_gradient (BSPLINE_Xform* bxf, BSPLINE_Score* ssd, char* fn);
void
dump_coeff (BSPLINE_Xform* bxf, char* fn);
void
dump_luts (BSPLINE_Xform* bxf);
void
dump_hist (BSPLINE_MI_Hist* mi_hist, char* fn);
void
bspline_save_debug_state 
(
 BSPLINE_Parms *parms, 
 Bspline_state *bst, 
 BSPLINE_Xform* bxf
 );

#if defined __cplusplus
}
#endif

#endif
