/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _bspline_h_
#define _bspline_h_

#include "plm_config.h"
#include "volume.h"

#define DOUBLE_HISTS	// Use doubles for histogram accumulation

/* -----------------------------------------------------------------------
   Macros
   ----------------------------------------------------------------------- */
#define INDEX_OF(ijk, dim) \
    (((ijk[2] * dim[1] + ijk[1]) * dim[0]) + ijk[0])

#define COORDS_FROM_INDEX(ijk, idx, dim) \
	ijk[2] = idx / (dim[0] * dim[1]);	\
	ijk[1] = (idx - (ijk[2] * dim[0] * dim[1])) / dim[0];	\
	ijk[0] = idx - ijk[2] * dim[0] * dim[1] - (ijk[1] * dim[0]);

/* -----------------------------------------------------------------------
   Types
   ----------------------------------------------------------------------- */
struct bspline_landmarks;

enum BsplineOptimization {
    BOPT_LBFGSB,
    BOPT_STEEPEST,
    BOPT_LIBLBFGS,
    BOPT_NLOPT_LBFGS,
    BOPT_NLOPT_LD_MMA,
    BOPT_NLOPT_PTN_1,
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

typedef struct Bspline_xform_struct Bspline_xform;
struct Bspline_xform_struct {
    float img_origin[3];         /* Image origin (in mm) */
    float img_spacing[3];        /* Image spacing (in mm) */
    int img_dim[3];              /* Image size (in vox) */
    int roi_offset[3];	         /* Position of first vox in ROI (in vox) */
    int roi_dim[3];		 /* Dimension of ROI (in vox) */
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

    float* q_dxdyz_lut;          /* Lookup table for influence of dN1/dx*dN2/dy*N3 */
    float* q_xdydz_lut;          /* Lookup table for influence of N1*dN2/dy*dN3/dz */
    float* q_dxydz_lut;          /* Lookup table for influence of dN1/dx*N2*dN3/dz */
    float* q_d2xyz_lut;          /* Lookup table for influence of (d2N1/dx2)*N2*N3 */
    float* q_xd2yz_lut;          /* Lookup table for influence of N1*(d2N2/dy2)*N3 */
    float* q_xyd2z_lut;          /* Lookup table for influence of N1*N2*(d2N3/dz2) */
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
    float* moving_image;	// Moving Image Voxels
    float* moving_grad;		// dc_dp (Gradient) Volume

    float* coeff;		// B-Spline coefficients (p)
    float* score;		// The "Score"

    float* f_hist_seg;		// "Segmented" fixed histogram
    float* m_hist_seg;		// "Segmented" moving histogram
    float* j_hist_seg;		// "Segmented" joint histogram

    float* f_hist;		// fixed image histogram
    float* m_hist;		// moving image histogram
    float* j_hist;		// joint histogram

    float* dc_dv;		// dc_dv (Interleaved)
    float* dc_dv_x;		// dc_dv (De-Interleaved)
    float* dc_dv_y;		// dc_dv (De-Interleaved)
    float* dc_dv_z;		// dc_dv (De-Interleaved)

    float* cond_x;		// dc_dv_x (Condensed)
    float* cond_y;		// dc_dv_y (Condensed)
    float* cond_z;		// dc_dv_z (Condensed)

    float* grad;		// dc_dp
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
    float* skipped;		// # of voxels that fell outside post warp

    int* c_lut;
    float* q_lut;


    // Zero Page Host Pointers
    float* zph_fixed_image;

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
    size_t skipped_size;
    size_t f_hist_size;
    size_t m_hist_size;
    size_t j_hist_size;
    size_t f_hist_seg_size;
    size_t m_hist_seg_size;
    size_t j_hist_seg_size;
    size_t c_lut_size;
    size_t q_lut_size;
};


typedef struct bspline_state Bspline_state;
struct bspline_state {
    int it;                              /* Number of iterations */
    int feval;                           /* Number of function evaluations */
    BSPLINE_Score ssd;                   /* Score and Gradient  */
    Dev_Pointers_Bspline* dev_ptrs;      /* GPU Device Pointers */
};

typedef struct BSPLINE_MI_Hist_Parms_struct BSPLINE_MI_Hist_Parms;
struct BSPLINE_MI_Hist_Parms_struct {
    long bins;
    float offset;
    float delta;
    int big_bin;    // fullest bin
};

typedef struct BSPLINE_MI_Hist_struct BSPLINE_MI_Hist;
struct BSPLINE_MI_Hist_struct {
    BSPLINE_MI_Hist_Parms moving;
    BSPLINE_MI_Hist_Parms fixed;
    BSPLINE_MI_Hist_Parms joint;    // JAS: for big_bin
    double* m_hist;
    double* f_hist;
    double* j_hist;
};

typedef struct Bspline_parms_struct Bspline_parms;
struct Bspline_parms_struct {
    enum BsplineThreading threading;
    enum BsplineOptimization optimization;
    enum BsplineMetric metric;
    int max_its;                 /* Max iterations (line searches) */
    int max_feval;               /* Max function evaluations */
    int debug;                   /* Create grad & histogram files */
    char implementation;         /* Implementation ('a', 'b', etc.) */
    int gpuid;                   /* Sets GPU to use for multi-gpu machines */
    double convergence_tol;      /* When to stop iterations based on score */
    int convergence_tol_its;     /* How many iterations to check for convergence tol */
    BSPLINE_MI_Hist mi_hist;     /* Histogram for MI score */
    void *data_on_gpu;           /* Pointer to structure encapsulating the data stored on the GPU */
    void *data_from_gpu;         /* Pointer to structure that stores the data returned from the GPU */
    char *xpm_hist_dump;         /* Pointer to base string of hist dumps */
    double lbfgsb_factr;         /* Function value tolerance for L-BFGS-B */
    double lbfgsb_pgtol;         /* Projected grad tolerance for L-BFGS-B */

    struct bspline_landmarks* landmarks;  /* The landmarks themselves */
    float landmark_stiffness;    /* Attraction of landmarks (0 == no attraction) */
    char landmark_implementation; /*Landmark score implementation, 'a' or 'b' */

    float young_modulus;  /* Penalty for having large gradient of the vector field */
    float rbf_radius;   /* Radius of RBF; if rbf_radius>0, RBF are used */
	float rbf_young_modulus; /* Penalty for the large 2nd derivative of RBF vector field */
};

/* -----------------------------------------------------------------------
   Function declarations
   ----------------------------------------------------------------------- */
#if defined __cplusplus
extern "C" {
#endif
gpuit_EXPORT
void bspline_parms_set_default (Bspline_parms* parms);

gpuit_EXPORT
void bspline_xform_set_default (Bspline_xform* bxf);

gpuit_EXPORT
Bspline_state *
bspline_state_create (
    Bspline_xform *bxf, 
    Bspline_parms *parms, 
    Volume *fixed, 
    Volume *moving, 
    Volume *moving_grad);

gpuit_EXPORT
void
bspline_xform_initialize (
    Bspline_xform* bxf,	         /* Output: bxf is initialized */
    float img_origin[3],         /* Image origin (in mm) */
    float img_spacing[3],        /* Image spacing (in mm) */
    int img_dim[3],              /* Image size (in vox) */
    int roi_offset[3],	         /* Position of first vox in ROI (in vox) */
    int roi_dim[3],		 /* Dimension of ROI (in vox) */
    int vox_per_rgn[3]);	 /* Knot spacing (in vox) */

gpuit_EXPORT
void bspline_xform_free (Bspline_xform* bxf);

gpuit_EXPORT
void bspline_parms_free (Bspline_parms* parms);

gpuit_EXPORT
void
bspline_state_destroy (Bspline_state *bst);

gpuit_EXPORT
void
bspline_warp (
    Volume *vout,         /* Output image (already sized and allocated) */
    Volume *vf_out,       /* Output vf (already sized and allocated, can be null) */
    Bspline_xform* bxf,   /* Bspline transform coefficients */
    Volume *moving,       /* Input image */
    int linear_interp,    /* 1 = trilinear, 0 = nearest neighbors */
    float default_val     /* Fill in this value outside of image */
);

gpuit_EXPORT
void bspline_run_optimization (
    Bspline_xform* bxf, 
    Bspline_state **bst,
    Bspline_parms *parms, 
    Volume *fixed, 
    Volume *moving, 
    Volume *moving_grad);
gpuit_EXPORT
Bspline_xform* read_bxf (char* filename);

gpuit_EXPORT
void write_bxf (char* filename, Bspline_xform* bxf);

gpuit_EXPORT
void
bspline_interpolate_vf (Volume* interp, 
			Bspline_xform* bxf);

/* Used internally */
void
bspline_interp_pix (float out[3], Bspline_xform *bxf, int p[3], int qidx);
void
bspline_interp_pix_b (
    float out[3], 
    Bspline_xform* bxf, 
    int pidx, 
    int qidx
);
int
bspline_find_correspondence 
(
 float *mxyz,             /* Output: xyz coordinates in moving image (mm) */
 float *mijk,             /* Output: ijk indices in moving image (vox) */
 const float *fxyz,       /* Input:  xyz coordinates in fixed image (mm) */
 const float *dxyz,       /* Input:  displacement from fixed to moving (mm) */
 const Volume *moving     /* Input:  moving image */
);

void
bspline_update_grad (
    Bspline_state *bst, 
    Bspline_xform* bxf, 
    int p[3], int qidx, float dc_dv[3]);

gpuit_EXPORT
void
bspline_initialize_mi (Bspline_parms* parms, Volume* fixed, Volume* moving);

void
bspline_set_coefficients (Bspline_xform* bxf, float val);

void
bspline_display_coeff_stats (Bspline_xform* bxf);

gpuit_EXPORT
void
bspline_score (Bspline_parms *parms, 
	       Bspline_state *bst,
	       Bspline_xform* bxf, 
	       Volume *fixed, 
	       Volume *moving, 
	       Volume *moving_grad);

void
clamp_linear_interpolate (float ma, int dmax, int* maf, int* mar, 
			  float* fa1, float* fa2);

void
bspline_update_grad_b (Bspline_state* bst, Bspline_xform* bxf, 
		       int pidx, int qidx, float dc_dv[3]);
int* calc_offsets (int* tile_dims, int* cdims);

void find_knots (int* knots, int tile_num, int* rdims, int* cdims);
void
dump_hist (BSPLINE_MI_Hist* mi_hist, int it);

void
report_score (char *alg, Bspline_xform *bxf, 
	      Bspline_state *bst, int num_vox, double timing);

/* Debugging routines */
void
dump_gradient (Bspline_xform* bxf, BSPLINE_Score* ssd, char* fn);

void
dump_coeff (Bspline_xform* bxf, char* fn);

void
dump_luts (Bspline_xform* bxf);

void
bspline_save_debug_state 
(
 Bspline_parms *parms, 
 Bspline_state *bst, 
 Bspline_xform* bxf
 );

void dump_xpm_hist (BSPLINE_MI_Hist* mi_hist, char* file_base, int iter);


#if defined __cplusplus
}
#endif

#endif
