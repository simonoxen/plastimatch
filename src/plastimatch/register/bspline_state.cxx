/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plmregister_config.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <math.h>
#ifndef _WIN32
#include <dlfcn.h>
#endif

#include "bspline.h"
#if (CUDA_FOUND)
#include "bspline_cuda.h"
#include "cuda_util.h"
#endif
#include "bspline_interpolate.h"
#include "bspline_landmarks.h"
#include "bspline_mi.h"
#include "bspline_mse.h"
#include "bspline_parms.h"
#include "bspline_regularize.h"
#include "bspline_state.h"
#include "bspline_xform.h"
#include "delayload.h"
#include "file_util.h"
#include "interpolate_macros.h"
#include "joint_histogram.h"
#include "logfile.h"
#include "plm_math.h"
#include "string_util.h"
#include "similarity_metric_type.h"
#include "volume.h"
#include "volume_macros.h"

static void
bspline_cuda_state_create (
    Bspline_parms *parms,
    Bspline_state *bst,
    Bspline_xform* bxf
);
static void
bspline_cuda_state_destroy (
    Bspline_parms *parms,
    Bspline_state *bst,
    Bspline_xform* bxf
);

class Bspline_state_private 
{
public:
    Bspline_parms *parms;
    Bspline_xform *bxf;
public:
    Bspline_state_private () {
        parms = 0;
        bxf = 0;
    }
    ~Bspline_state_private () {
        /* Members not owned by this class */
    }
};


Bspline_state::Bspline_state ()
{
    d_ptr = new Bspline_state_private;
    mi_hist = 0;
}

Bspline_state::~Bspline_state ()
{
    bspline_cuda_state_destroy (d_ptr->parms, this, d_ptr->bxf);
    delete d_ptr;
    delete mi_hist;
}

void
Bspline_state::initialize (
    Bspline_xform *bxf,
    Bspline_parms *parms)
{
    Regularization_parms* reg_parms = parms->reg_parms;
    Bspline_regularize* rst = &this->rst;
    Bspline_landmarks* blm = parms->blm;

    d_ptr->bxf = bxf;
    d_ptr->parms = parms;

    this->sm = 0;
    this->it = 0;
    this->feval = 0;
    this->dev_ptrs = 0;
    this->mi_hist = 0;

    this->ssd.set_num_coeff (bxf->num_coeff);

    if (reg_parms->lambda > 0.0f) {
        rst->fixed_stiffness = parms->fixed_stiffness;
        rst->initialize (reg_parms, bxf);
    }

    /* Initialize MI histograms */
    if (parms->has_metric_type (SIMILARITY_METRIC_MI_MATTES)) {
        this->mi_hist = new Joint_histogram (
            parms->mi_hist_type,
            parms->mi_hist_fixed_bins,
            parms->mi_hist_moving_bins);
    }

    /* Landmarks */
    blm->initialize (bxf);
}

void
Bspline_state::initialize_similarity_images ()
{
    /* GCS FIX: The below function also does other initializations 
       which do not require the similarity images, and therefore could 
       be done once per stage rather than once per image
     */
    /* Copy images into CUDA memory */
    bspline_cuda_state_create (d_ptr->parms, this, d_ptr->bxf);
}

static void
bspline_cuda_state_create (
    Bspline_parms *parms,
    Bspline_state *bst,
    Bspline_xform *bxf
)
{
#if (CUDA_FOUND)
    if (parms->threading != BTHR_CUDA) {
        return;
    }
    if (bst->dev_ptrs) {
        bspline_cuda_state_destroy (parms, bst, bxf);
    }

    /* Set the gpuid */
    LOAD_LIBRARY_SAFE (libplmcuda);
    LOAD_SYMBOL (CUDA_selectgpu, libplmcuda);
    CUDA_selectgpu (parms->gpuid);
    UNLOAD_LIBRARY (libplmcuda);
    
    Volume *fixed = bst->fixed;
    Volume *moving = bst->moving;
    Volume *moving_grad = bst->moving_grad;

    Dev_Pointers_Bspline* dev_ptrs 
        = (Dev_Pointers_Bspline*) malloc (sizeof (Dev_Pointers_Bspline));
    bst->dev_ptrs = dev_ptrs;

    /* GCS FIX: You cannot have more than one CUDA metric because 
       dev_ptrs is not defined per metric */
    if (parms->has_metric_type (SIMILARITY_METRIC_MSE)) {
        /* Be sure we loaded the CUDA plugin */
        LOAD_LIBRARY_SAFE (libplmregistercuda);
        LOAD_SYMBOL (CUDA_bspline_mse_init_j, libplmregistercuda);

        switch (parms->implementation) {
        case 'j':
        case '\0':   /* Default */
            CUDA_bspline_mse_init_j (dev_ptrs, fixed, moving, moving_grad, bxf, parms);
            break;
        default:
            printf ("Warning: option -f %c unavailble.  Switching to -f j\n",
                parms->implementation);
            CUDA_bspline_mse_init_j (dev_ptrs, fixed, moving, moving_grad, bxf, parms);
            break;
        }

        UNLOAD_LIBRARY (libplmregistercuda);
    } 
    else if (parms->has_metric_type (SIMILARITY_METRIC_MI_MATTES)) {
        /* Be sure we loaded the CUDA plugin */
        LOAD_LIBRARY_SAFE (libplmregistercuda);
        LOAD_SYMBOL (CUDA_bspline_mi_init_a, libplmregistercuda);

        switch (parms->implementation) {
        case 'a':
            CUDA_bspline_mi_init_a (bxf, bst, dev_ptrs, fixed, moving, moving_grad);
            break;
        default:
            printf ("Warning: option -f %c unavailble.  Defaulting to -f a\n",
                parms->implementation);
            CUDA_bspline_mi_init_a (bxf, bst, dev_ptrs, fixed, moving, moving_grad);
            break;
        }

        UNLOAD_LIBRARY (libplmregistercuda);
    }
    else {
        printf ("No cuda initialization performed.\n");
    }
#endif
}

static void
bspline_cuda_state_destroy (
    Bspline_parms *parms, 
    Bspline_state *bst,
    Bspline_xform *bxf
)
{
#if (CUDA_FOUND)
    if (parms->threading != BTHR_CUDA) {
        return;
    }

    Volume *fixed = bst->fixed;
    Volume *moving = bst->moving;
    Volume *moving_grad = bst->moving_grad;

    if (parms->has_metric_type (SIMILARITY_METRIC_MSE)) {
        LOAD_LIBRARY_SAFE (libplmregistercuda);
        LOAD_SYMBOL (CUDA_bspline_mse_cleanup_j, libplmregistercuda);
        CUDA_bspline_mse_cleanup_j ((Dev_Pointers_Bspline *) bst->dev_ptrs, fixed, moving, moving_grad);
        UNLOAD_LIBRARY (libplmregistercuda);
    }
    else if (parms->has_metric_type (SIMILARITY_METRIC_MI_MATTES)) {
        LOAD_LIBRARY_SAFE (libplmregistercuda);
        LOAD_SYMBOL (CUDA_bspline_mi_cleanup_a, libplmregistercuda);
        CUDA_bspline_mi_cleanup_a ((Dev_Pointers_Bspline *) bst->dev_ptrs, fixed, moving, moving_grad);
        UNLOAD_LIBRARY (libplmregistercuda);
    }

    free (bst->dev_ptrs);
    bst->dev_ptrs = 0;
#endif
}
