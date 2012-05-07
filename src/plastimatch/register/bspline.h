/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _bspline_h_
#define _bspline_h_

#include "plmregister_config.h"
#include "sys/plm_int.h"
#include <string>

#include "bspline_landmarks.h"
#include "bspline_regularize.h"
#include "bspline_regularize_state.h"

#define DOUBLE_HISTS	// Use doubles for histogram accumulation

/* JAS 2011.07.23
 * The following is a fix that allows us to more selectively enforce
 * the -malign-double compatibility required by object files compiled
 * by nvcc.  Any structures that are used by both nvcc compiled files
 * and gcc/g++ compiled files should use this.  The reason we do not
 * simply pass -malign-double to gcc/g++ in order to achieve this
 * compatibility is because Slicer's ITK does not come compiled with
 * the -malign-double flag on 32-bit systems... so, believe it or not
 * this might be the cleanest solution */
#if (__GNUC__) && (MACHINE_IS_32_BIT) && (CUDA_FOUND)
    typedef double double_align8 __attribute__ ((aligned(8)));
#else 
    typedef double double_align8;
#endif

/* -----------------------------------------------------------------------
   Types
   ----------------------------------------------------------------------- */
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
    BTHR_CUDA
};

enum BsplineMetric {
    BMET_MSE,
    BMET_MI
};

enum BsplineHistType {
    HIST_EQSP,
    HIST_VOPT
};

class Bspline_score {
public:
    float score;         /* Total Score (sent to optimizer) */
    float lmetric;       /* Landmark metric */
    float rmetric;       /* Regularization metric */
    float smetric;       /* Similarity metric */
    int num_vox;         /* Number of voxel with correspondence */

    float* grad;         /* Gradient score wrt control coeff */

    double time_smetric;   /* Time to compute similarity metric */
    double time_rmetric;   /* Time to compute regularization metric */
public:
    Bspline_score () {
        this->score = 0;
        this->lmetric = 0;
        this->rmetric = 0;
        this->smetric = 0;
        this->num_vox = 0;
        this->grad = 0;

        this->time_smetric = 0;
        this->time_rmetric = 0;
    }
};

class Bspline_state {
public:
    int it;              /* Number of iterations */
    int feval;           /* Number of function evaluations */
    Bspline_score ssd;   /* Score and Gradient  */
    void* dev_ptrs;      /* GPU Device Pointers */
    Bspline_regularize_state rst;       /* Analytic regularization */
};

class Bspline_mi_hist_Parms {
public:
    /* Used by all histogram types */
    enum BsplineHistType type;  /* Type of histograms */
    plm_long bins;           /* # of bins in histogram  */
    float offset;               /* minimum voxel intensity */
    plm_long big_bin;             /* fullest bin index       */
    float delta;                /* bin OR key spacing   */

    /* For V-Optimal Histograms */
    plm_long keys;                /* # of keys               */
    int* key_lut;               /* bin keys lookup table   */
};

class Bspline_mi_hist {
public:
    Bspline_mi_hist_Parms moving;
    Bspline_mi_hist_Parms fixed;
    Bspline_mi_hist_Parms joint;    // JAS: for big_bin
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
    double_align8 convergence_tol; /* When to stop iterations based on score */
    int convergence_tol_its;     /* How many iterations to check for 
				    convergence tol */
    Bspline_mi_hist mi_hist;     /* Histogram for MI score */
    void *data_on_gpu;           /* Pointer to structure encapsulating the 
				    data stored on the GPU */
    void *data_from_gpu;         /* Pointer to structure that stores the 
				    data returned from the GPU */
    double_align8 lbfgsb_factr;  /* Function value tolerance for L-BFGS-B */
    double_align8 lbfgsb_pgtol;  /* Projected grad tolerance for L-BFGS-B */

    /* Image Volumes */
    Volume *fixed;
    Volume *moving;
    Volume *moving_grad;

    /* Regularization */
    Reg_parms reg_parms;         /* Regularization Parameters */

    /* Landmarks */
    Bspline_landmarks blm;       /* Landmarks parameters */

    float rbf_radius;            /* Radius of RBF; if rbf_radius>0, RBF 
				    are used */
    float rbf_young_modulus;     /* Penalty for the large 2nd derivative 
				    of RBF vector field */
    char *xpm_hist_dump;         /* Pointer to base string of hist dumps */

    /* JAS - temporary home for these */
    Volume* fixed_mask;
    Volume* moving_mask;

public:
    Bspline_parms () {
	this->threading = BTHR_CPU;
	this->optimization = BOPT_LBFGSB;
	this->metric = BMET_MSE;
	this->implementation = '\0';
	this->max_its = 10;
	this->max_feval = 10;
	this->debug = 0;
	this->debug_dir = ".";
	this->debug_stage = 0;
	this->gpuid = 0;
	this->convergence_tol = 0.1;
	this->convergence_tol_its = 4;
	this->mi_hist.f_hist = 0;
	this->mi_hist.m_hist = 0;
	this->mi_hist.j_hist = 0;
	this->mi_hist.fixed.type = HIST_EQSP;
	this->mi_hist.moving.type = HIST_EQSP;
	this->mi_hist.fixed.bins = 32;
	this->mi_hist.moving.bins = 32;
	this->mi_hist.joint.bins 
	    = this->mi_hist.fixed.bins * this->mi_hist.moving.bins;
	this->mi_hist.fixed.big_bin = 0;
	this->mi_hist.moving.big_bin = 0;
	this->mi_hist.joint.big_bin = 0;
	this->data_on_gpu = 0;
	this->data_from_gpu = 0;
	this->lbfgsb_factr = 1.0e+7;
	this->lbfgsb_pgtol = 1.0e-5;

	this->rbf_radius = 0;
	this->rbf_young_modulus = 0;
	this->xpm_hist_dump = 0;
    this->fixed_mask = NULL;
    this->moving_mask = NULL;
    }
};

class Bspline_optimize_data {
public:
    Bspline_xform* bxf;
    Bspline_state *bst;
    Bspline_parms *parms;
    Volume *fixed;
    Volume *moving;
    Volume *moving_grad;
};


C_API Bspline_state* bspline_state_create (
    Bspline_xform *bxf, 
    Bspline_parms *parms
);
C_API void bspline_parms_free (Bspline_parms* parms);
C_API void bspline_state_destroy (
    Bspline_state *bst,
    Bspline_parms *parms,
    Bspline_xform *bxf
);
C_API Volume* bspline_compute_vf (const Bspline_xform* bxf);
void bspline_display_coeff_stats (Bspline_xform* bxf);
C_API void bspline_score (Bspline_optimize_data *bod);
void bspline_update_grad (
    Bspline_state *bst, 
    Bspline_xform* bxf, 
    plm_long p[3], plm_long qidx, float dc_dv[3]
);
void bspline_update_grad_b (
    Bspline_score* bscore,
    const Bspline_xform* bxf, 
    plm_long pidx, 
    plm_long qidx, 
    const float dc_dv[3]
);
int* calc_offsets (int* tile_dims, int* cdims);
void find_knots (plm_long* knots, plm_long tile_num, plm_long* rdims, plm_long* cdims);
void dump_hist (Bspline_mi_hist* mi_hist, int it);
void report_score (
    Bspline_parms *parms,
    Bspline_xform *bxf, 
    Bspline_state *bst
);

/* Debugging routines */
void dump_gradient (Bspline_xform* bxf, Bspline_score* ssd, char* fn);
void bspline_save_debug_state (
    Bspline_parms *parms, 
    Bspline_state *bst, 
    Bspline_xform* bxf
);
void dump_xpm_hist (Bspline_mi_hist* mi_hist, char* file_base, int iter);
void bspline_make_grad (
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
