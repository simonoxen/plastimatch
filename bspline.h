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

enum BsplineImplementation {
    BIMPL_CPU,
    BIMPL_BROOK,
	BIMPL_CUDA
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
    int roi_offset[3];		     /* Position of first vox in ROI (in vox) */
    int roi_dim[3];		         /* Dimension of ROI (in vox) */
    int vox_per_rgn[3];		     /* Knot spacing (in vox) */
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

typedef struct BSPLINE_MI_Hist_Parms_struct BSPLINE_MI_Hist_Parms;
struct BSPLINE_MI_Hist_Parms_struct {
    int bins;
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
    enum BsplineImplementation implementation;
    enum BsplineOptimization optimization;
    enum BsplineMetric metric;
    int max_its;
    int debug;			 /* Create grad & histogram files */
#if defined (commentout)
    float img_origin[3];         /* Image origin (in mm) */
    float img_spacing[3];        /* Image spacing (in mm) */
    int img_dim[3];              /* Image size (in vox) */
    int roi_offset[3];		 /* Position of first vox in ROI (in vox) */
    int roi_dim[3];		 /* Dimension of ROI (in vox) */
    int vox_per_rgn[3];		 /* Knot spacing (in vox) */
    float grid_spac[3];          /* Knot spacing (in mm) */
#endif
    BSPLINE_MI_Hist mi_hist;     /* Histogram for MI score */
    BSPLINE_Score ssd;           /* Score and Gradient */
    void *data_on_gpu;		 /* Pointer to structure encapsulating the data stored on the GPU */
    void *data_from_gpu;	 /* Pointer to structure that stores the data returned from the GPU */
};

#if defined __cplusplus
extern "C" {
#endif
void bspline_parms_set_default (BSPLINE_Parms* parms);
void bspline_xform_set_default (BSPLINE_Xform* bxf);
void bspline_xform_initialize (BSPLINE_Xform* bxf, 
		    float img_origin[3],         /* Image origin (in mm) */
		    float img_spacing[3],        /* Image spacing (in mm) */
		    int img_dim[3],              /* Image size (in vox) */
		    int roi_offset[3],		 /* Position of first vox in ROI (in vox) */
		    int roi_dim[3],		 /* Dimension of ROI (in vox) */
		    int vox_per_rgn[3]);	 /* Knot spacing (in vox) */
void bspline_xform_free (BSPLINE_Xform* bxf);
void bspline_parms_free (BSPLINE_Parms* parms);
void bspline_optimize (BSPLINE_Xform* bxf, BSPLINE_Parms *parms, Volume *fixed, Volume *moving, 
		  Volume *moving_grad);
void write_bxf (char* filename, BSPLINE_Xform* bxf);

/* Used internally */
void
bspline_set_coefficients (BSPLINE_Xform* bxf, float val);
void
bspline_display_coeff_stats (BSPLINE_Xform* bxf);
void
bspline_score (BSPLINE_Parms* parms, BSPLINE_Xform* bxf, Volume *fixed, Volume *moving, 
	       Volume *moving_grad);
void
bspline_score_reference (BSPLINE_Score *ssd, 
			 Volume *fixed, Volume *moving, Volume *moving_grad, 
			 BSPLINE_Parms *parms);
void
bspline_interpolate_vf (Volume* interp, 
			BSPLINE_Xform* bxf);

void
clamp_linear_interpolate(float ma, int dmax, int* maf, int* mar, float* fa1, float* fa2);

void
bspline_update_grad_b (BSPLINE_Parms* parms, BSPLINE_Xform* bxf, int pidx, int qidx, float dc_dv[3]);

#if defined __cplusplus
}
#endif

#endif
