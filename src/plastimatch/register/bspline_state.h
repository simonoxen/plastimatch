/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _bspline_state_h_
#define _bspline_state_h_

#include "plmregister_config.h"
#include <string>

#include "bspline_mi_hist.h"
#include "bspline_regularize.h"
#include "bspline_score.h"
#include "plm_int.h"
#include "smart_pointer.h"
#include "stage_similarity_data.h"

class Bspline_state_private;
class Bspline_parms;

class PLMREGISTER_API Bspline_state {
public:
    SMART_POINTER_SUPPORT (Bspline_state);
    Bspline_state_private *d_ptr;
public:
    Bspline_state ();
    ~Bspline_state ();
    void initialize (Bspline_xform *bxf, Bspline_parms *parms);
public:
    int sm;                         /* Current smetric */
    int it;                         /* Current iterations */
    int feval;                      /* Number of function evaluations */
    Bspline_score ssd;              /* Score and Gradient  */
    void* dev_ptrs;                 /* GPU Device Pointers */

    /*! \brief Current similarity images */
    /* GCS FIX.  These can be replaced with Stage_similarity_data 
       if nvcc can be made to use a c++ compiler */
    Volume *fixed;
    Volume *moving;
    Volume *moving_grad;
    Volume *fixed_roi;
    Volume *moving_roi;
    
    Bspline_regularize rst;         /* Analytic regularization */
    Bspline_mi_hist_set *mi_hist;   /* MI histograms */
public:
    Bspline_score* get_bspline_score () {
        return &ssd;
    }
};

#endif
