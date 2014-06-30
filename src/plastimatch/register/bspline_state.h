/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _bspline_state_h_
#define _bspline_state_h_

#include "plmregister_config.h"
#include "plm_int.h"
#include <string>

#include "bspline_mi_hist.h"
#include "bspline_regularize.h"
#include "bspline_score.h"

class Bspline_state {
public:
public:
    int it;              /* Number of iterations */
    int feval;           /* Number of function evaluations */
    Bspline_score ssd;   /* Score and Gradient  */
    void* dev_ptrs;      /* GPU Device Pointers */
    Bspline_regularize rst;   /* Analytic regularization */
    Bspline_mi_hist_set *mi_hist;   /* MI histograms */
public:
    Bspline_state ();
    ~Bspline_state ();
};

PLMREGISTER_C_API Bspline_state* bspline_state_create (
    Bspline_xform *bxf, 
    Bspline_parms *parms
);
PLMREGISTER_C_API void bspline_state_destroy (
    Bspline_state *bst,
    Bspline_parms *parms,
    Bspline_xform *bxf
);

#endif
