/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _bspline_h_
#define _bspline_h_

#include "plm_config.h"
#include <string>
#include "bspline_xform.h"
#include "volume.h"
#include "reg.h"

#define DOUBLE_HISTS	// Use doubles for histogram accumulation

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

typedef struct BSPLINE_Score_struct BSPLINE_Score;
struct BSPLINE_Score_struct {
    float score;
    float* grad;
};

typedef struct bspline_state Bspline_state;
struct bspline_state {
    int it;                              /* Number of iterations */
    int feval;                           /* Number of function evaluations */
    BSPLINE_Score ssd;                   /* Score and Gradient  */
    void* dev_ptrs;                      /* GPU Device Pointers */
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

class Bspline_parms
{
public:
    enum BsplineThreading threading;
    enum BsplineOptimization optimization;
    enum BsplineMetric metric;
    char implementation;         /* Implementation ('a', 'b', etc.) */
    int max_its;                 /* Max iterations (line searches) */
    int max_feval;               /* Max function evaluations */
    int debug;                   /* Create grad & histogram files */
    std::string debug_dir;       /* Directory where to create debug files */
    int debug_stage;             /* Used to tag debug files by stage */
    int gpuid;                   /* Sets GPU to use for multi-gpu machines */
    int gpu_zcpy;                /* Use zero-copy when possible? */
    double convergence_tol;      /* When to stop iterations based on score */
    int convergence_tol_its;     /* How many iterations to check for 
				    convergence tol */
    BSPLINE_MI_Hist mi_hist;     /* Histogram for MI score */
    void *data_on_gpu;           /* Pointer to structure encapsulating the 
				    data stored on the GPU */
    void *data_from_gpu;         /* Pointer to structure that stores the 
				    data returned from the GPU */
    double lbfgsb_factr;         /* Function value tolerance for L-BFGS-B */
    double lbfgsb_pgtol;         /* Projected grad tolerance for L-BFGS-B */

    struct bspline_landmarks* landmarks;  /* The landmarks themselves */
    float landmark_stiffness;    /* Attraction of landmarks (0 == no 
				    attraction) */
    char landmark_implementation; /*Landmark score implementation, 'a' or 'b' */

    float young_modulus;         /* Penalty for having large gradient 
				    of the vector field */
    float rbf_radius;            /* Radius of RBF; if rbf_radius>0, RBF 
				    are used */
    float rbf_young_modulus;     /* Penalty for the large 2nd derivative 
				    of RBF vector field */
    char *xpm_hist_dump;         /* Pointer to base string of hist dumps */
    Reg_parms reg_parms;         /* Regularization Parameters */
public:
    Bspline_parms () {
	this->threading = BTHR_CPU;
	this->optimization = BOPT_LBFGSB;
	this->metric = BMET_MSE;
	this->implementation = '\0';
	this->max_its = 10;
	this->max_feval = 10;
	this->debug = 0;
	this->debug_dir = "";
	this->debug_stage = 0;
	this->gpuid = 0;
	this->gpu_zcpy = 0;
	this->convergence_tol = 0.1;
	this->convergence_tol_its = 4;
	this->mi_hist.f_hist = 0;
	this->mi_hist.m_hist = 0;
	this->mi_hist.j_hist = 0;
	this->mi_hist.fixed.bins = 500;
	this->mi_hist.moving.bins = 500;
	this->mi_hist.joint.bins 
	    = this->mi_hist.fixed.bins * this->mi_hist.moving.bins;
	this->mi_hist.fixed.big_bin = 0;
	this->mi_hist.moving.big_bin = 0;
	this->mi_hist.joint.big_bin = 0;
	this->data_on_gpu = 0;
	this->data_from_gpu = 0;
	this->lbfgsb_factr = 1.0e+7;
	this->lbfgsb_pgtol = 1.0e-5;
	this->landmarks = 0;
	this->landmark_stiffness = 0;
	this->landmark_implementation = 'a';
	this->young_modulus = 0;
	this->rbf_radius = 0;
	this->rbf_young_modulus = 0;
	this->xpm_hist_dump = 0;
    }
};

/* -----------------------------------------------------------------------
   Function declarations
   ----------------------------------------------------------------------- */
#if defined __cplusplus
extern "C" {
#endif

gpuit_EXPORT
Bspline_state *
bspline_state_create (
    Bspline_xform *bxf, 
    Bspline_parms *parms, 
    Volume *fixed, 
    Volume *moving, 
    Volume *moving_grad);

gpuit_EXPORT
void bspline_parms_free (Bspline_parms* parms);

gpuit_EXPORT
void
bspline_state_destroy (
    Bspline_state *bst,
    Bspline_parms *parms,
    Volume* fixed,
    Volume* moving,
    Volume* moving_grad);

gpuit_EXPORT
void
bspline_transform_point (
    float point_out[3], /* Output coordinate of point */
    Bspline_xform* bxf, /* Bspline transform coefficients */
    float point_in[3],  /* Input coordinate of point */
    int linear_interp   /* 1 = trilinear, 0 = nearest neighbors */
);

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


gpuit_EXPORT
void
bspline_initialize_mi (Bspline_parms* parms, Volume* fixed, Volume* moving);

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
bspline_update_grad (
    Bspline_state *bst, 
    Bspline_xform* bxf, 
    int p[3], int qidx, float dc_dv[3]);
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
bspline_save_debug_state 
(
 Bspline_parms *parms, 
 Bspline_state *bst, 
 Bspline_xform* bxf
 );

void dump_xpm_hist (BSPLINE_MI_Hist* mi_hist, char* file_base, int iter);

void
bspline_make_grad (float* cond_x, float* cond_y, float* cond_z,
    Bspline_xform* bxf, BSPLINE_Score* ssd);

void
bspline_update_sets (float* sets_x, float* sets_y, float* sets_z,
    int qidx, float* dc_dv, Bspline_xform* bxf);

void
bspline_sort_sets (float* cond_x, float* cond_y, float* cond_z,
    float* sets_x, float* sets_y, float* sets_z,
    int pidx, Bspline_xform* bxf);

#if defined __cplusplus
}
#endif

#endif
