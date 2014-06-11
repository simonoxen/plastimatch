/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _bspline_parms_h_
#define _bspline_parms_h_

#include "plmregister_config.h"
#include <string>
#include "bspline_mi_hist.h"

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

class Bspline_landmarks;
class Reg_parms;

class PLMREGISTER_API Bspline_parms
{
public:
    enum BsplineThreading threading;
    enum BsplineOptimization optimization;
    enum BsplineMetric metric;
    char implementation;         /* Implementation ('a', 'b', etc.) */
    int min_its;                 /* Miniumum iterations (line searches) */
    int max_its;                 /* Max iterations (line searches) */
    int max_feval;               /* Max function evaluations */
    int debug;                   /* Create grad & histogram files */
    std::string debug_dir;       /* Directory where to create debug files */
    int debug_stage;             /* Used to tag debug files by stage */
    int gpuid;                   /* Sets GPU to use for multi-gpu machines */
    double_align8 convergence_tol; /* When to stop iterations based on score */

    /* MI parms */
    Bspline_mi_hist_type mi_hist_type;
    plm_long mi_hist_fixed_bins;
    plm_long mi_hist_moving_bins;

    float mi_fixed_image_minVal;
    float mi_fixed_image_maxVal;
    float mi_moving_image_minVal;
    float mi_moving_image_maxVal;

    /* LBFGSB optimizer parms */
    double_align8 lbfgsb_factr;  /* Function value tolerance for L-BFGS-B */
    double_align8 lbfgsb_pgtol;  /* Projected grad tolerance for L-BFGS-B */

    /* Image Volumes */
    Volume* fixed;               /* Not owned by Bspline_parms */
    Volume* moving;              /* Not owned by Bspline_parms */
    Volume* moving_grad;         /* Not owned by Bspline_parms */
    Volume* fixed_roi;           /* Not owned by Bspline_parms */
    Volume* moving_roi;          /* Not owned by Bspline_parms */

    /* Regularization */
    Reg_parms* reg_parms;        /* Regularization Parameters */

    /* Landmarks */
    Bspline_landmarks* blm;      /* Landmarks parameters */
    float rbf_radius;            /* Radius of RBF; if rbf_radius>0, RBF are used */
    float rbf_young_modulus;     /* Penalty for the large 2nd derivative of RBF vector field */
    char* xpm_hist_dump;         /* Pointer to base string of hist dumps */

public:
    Bspline_parms ();
    ~Bspline_parms ();
};

#endif
