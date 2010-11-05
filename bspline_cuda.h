/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _bspline_cuda_h_
#define _bspline_cuda_h_

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <math.h>
#include <cuda.h>
#include "volume.h"
#include "bspline.h"
#include "cuda_mem.h"

/* B-Spline CUDA MI Switches */
//#define MI_HISTS_CPU
//#define MI_GRAD_CPU
#define MI_SCORE_CPU

/* Uncomment to profile CUDA MSE */
//#define PROFILE_MSE

typedef struct dev_pointers_bspline Dev_Pointers_Bspline;
struct dev_pointers_bspline
{
    // IMPORTANT!
    // Each member of this struct is a POINTER TO
    // AN ADDRESS RESIDING IN THE GPU'S GLOBAL
    // MEMORY!  Care must be taken when referencing
    // and dereferencing members of this structure!

    float* fixed_image;     // Fixed Image Voxels
    float* moving_image;    // Moving Image Voxels
    float* moving_grad;     // dc_dp (Gradient) Volume

    float* coeff;           // B-Spline coefficients (p)
    float* score;           // The "Score"

    float* dc_dv_x;         // dc_dv (De-Interleaved)
    float* dc_dv_y;         // dc_dv (De-Interleaved)
    float* dc_dv_z;         // dc_dv (De-Interleaved)

    float* cond_x;          // dc_dv_x (Condensed)
    float* cond_y;          // dc_dv_y (Condensed)
    float* cond_z;          // dc_dv_z (Condensed)

    float* grad;            // dc_dp

    float* f_hist_seg;      // "Segmented" fixed histogram
    float* m_hist_seg;      // "Segmented" moving histogram
    float* j_hist_seg;      // "Segmented" joint histogram

    float* f_hist;          // fixed image histogram
    float* m_hist;          // moving image histogram
    float* j_hist;          // joint histogram

    int* LUT_Knot;          // Control Point LUT
    int* LUT_Offsets;       // Tile Offset LUT
    float* LUT_Bspline_x;   // Pre-computed Bspline evaluations
    float* LUT_Bspline_y;   // ------------ '' ----------------
    float* LUT_Bspline_z;   // ------------ '' ----------------

    // # of voxels that do not have a correspondence
    float* skipped;                 // Legacy (for GPU w/o atomics)
    unsigned int* skipped_atomic;   // New (for GPU w/ Global atomics)

    // Head of linked list tracking pinned CPU memory
    // NOTE: This is the only pointer in this struct containing
    //       a pointer from the CPU memory map.
    Vmem_Entry* vmem_list;

    // Sizes of allocations for above pointers.
    size_t fixed_image_size;
    size_t moving_image_size;
    size_t moving_grad_size;

    size_t coeff_size;
    size_t score_size;

    size_t dc_dv_x_size;
    size_t dc_dv_y_size;
    size_t dc_dv_z_size;

    size_t cond_x_size;
    size_t cond_y_size;
    size_t cond_z_size;

    size_t grad_size;

    size_t f_hist_seg_size;
    size_t m_hist_seg_size;
    size_t j_hist_seg_size;

    size_t f_hist_size;
    size_t m_hist_size;
    size_t j_hist_size;

    size_t LUT_Knot_size;
    size_t LUT_Offsets_size;
    size_t LUT_Bspline_x_size;
    size_t LUT_Bspline_y_size;
    size_t LUT_Bspline_z_size;
    size_t skipped_size;
};

#if defined __cplusplus
extern "C" {
#endif

    // -------------------------------------------------------------------
    // Prototypes: bspline_cuda.cpp 

    void
    bspline_cuda_MI_a (
        Bspline_parms *parms,
        Bspline_state *bst,
        Bspline_xform *bxf,
        Volume *fixed,
        Volume *moving,
        Volume *moving_grad,
        Dev_Pointers_Bspline *dev_ptrs);

    void
    bspline_cuda_score_j_mse (
        Bspline_parms* parms,
        Bspline_state *bst,
        Bspline_xform* bxf,
        Volume* fixed,
        Volume* moving,
        Volume* moving_grad,
        Dev_Pointers_Bspline* dev_ptrs
    );

    //
    // -------------------------------------------------------------------




    // -------------------------------------------------------------------
    // Prototypes: bspline_cuda.cu

    void
    bspline_cuda_j_stage_1 (
        Volume* fixed,
        Volume* moving,
        Volume* moving_grad,
        Bspline_xform* bxf,
        Bspline_parms* parms,
        Dev_Pointers_Bspline* dev_ptrs
    );

    void
    bspline_cuda_j_stage_2 (
        Bspline_parms* parms,
        Bspline_xform* bxf,
        Volume* fixed,
        int* vox_per_rgn,
        int* volume_dim,
        float* host_score,
        float* host_grad,
        float* host_grad_mean,
        float* host_grad_norm,
        Dev_Pointers_Bspline* dev_ptrs,
        int* num_vox
    );

    void
    bspline_cuda_initialize_j (
        Dev_Pointers_Bspline* dev_ptrs,
        Volume* fixed,
        Volume* moving,
        Volume* moving_grad,
        Bspline_xform* bxf,
        Bspline_parms* parms
    );

    void
    bspline_cuda_clean_up_mse_j (
        Dev_Pointers_Bspline* dev_ptrs,
        Volume* fixed,
        Volume* moving,
        Volume* moving_grad
    );

    void
    bspline_cuda_clean_up_mi_a (
        Dev_Pointers_Bspline* dev_ptrs,
        Volume* fixed,
        Volume* moving,
        Volume* moving_grad
    );

    void
    CUDA_bspline_push_coeff (
        Dev_Pointers_Bspline* dev_ptrs,
        Bspline_xform* bxf
    );
    
    void
    CUDA_bspline_zero_score (
        Dev_Pointers_Bspline* dev_ptrs
    );
    
    void
    CUDA_bspline_zero_grad (
        Dev_Pointers_Bspline* dev_ptrs
    );

    void
    CUDA_bspline_mse_score_dc_dv (
        Dev_Pointers_Bspline* dev_ptrs,
        Bspline_xform* bxf,
        Volume* fixed,
        Volume* moving
    );

    void
    CUDA_bspline_condense (
        Dev_Pointers_Bspline* dev_ptrs,
        int* vox_per_rgn,
        int num_tiles
    );

    void
    CUDA_bspline_reduce (
        Dev_Pointers_Bspline* dev_ptrs,
        int num_knots
    );

    float
    CPU_obtain_spline_basis_function (
        int t_idx,
        int vox_idx,
        int vox_per_rgn
    ); 
    
    void
    bspline_cuda_init_MI_a (
        Dev_Pointers_Bspline* dev_ptrs,
        Volume* fixed,
        Volume* moving,
        Volume* moving_grad,
        Bspline_xform* bxf,
        Bspline_parms* parms
    );

    int
    CUDA_bspline_MI_a_hist (
        Dev_Pointers_Bspline *dev_ptrs,
        BSPLINE_MI_Hist* mi_hist,
        Volume* fixed,
        Volume* moving,
        Bspline_xform *bxf
    );

    void
    CUDA_bspline_MI_a_hist_fix (
        Dev_Pointers_Bspline *dev_ptrs,
        BSPLINE_MI_Hist* mi_hist,
        Volume* fixed,
        Volume* moving,
        Bspline_xform *bxf
    );
    
    void
    CUDA_bspline_MI_a_hist_mov (
        Dev_Pointers_Bspline *dev_ptrs,
        BSPLINE_MI_Hist* mi_hist,
        Volume* fixed,
        Volume* moving,
        Bspline_xform *bxf
    );
    
    int
    CUDA_bspline_MI_a_hist_jnt (
        Dev_Pointers_Bspline *dev_ptrs,
        BSPLINE_MI_Hist* mi_hist,
        Volume* fixed,
        Volume* moving,
        Bspline_xform *bxf
    );

    int
    CUDA_MI_Hist_a (
        BSPLINE_MI_Hist* mi_hist,
        Bspline_xform *bxf,
        Volume* fixed,
        Volume* moving,
        Dev_Pointers_Bspline *dev_ptrs
    );
    
    void
    CUDA_MI_Grad_a (
        BSPLINE_MI_Hist* mi_hist,
        Bspline_state *bst,
        Bspline_xform *bxf,
        Volume* fixed,
        Volume* moving,
        float num_vox_f,
        Dev_Pointers_Bspline *dev_ptrs
    );

    //
    // -------------------------------------------------------------------

#if defined __cplusplus
}
#endif

#endif
