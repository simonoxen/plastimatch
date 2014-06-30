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
    int it;              /* Number of iterations */
    int feval;           /* Number of function evaluations */
    Bspline_score ssd;   /* Score and Gradient  */
    void* dev_ptrs;      /* GPU Device Pointers */
    Bspline_regularize rst;   /* Analytic regularization */
    Bspline_mi_hist_set *mi_hist;   /* MI histograms */
};

#if defined (commentout)
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

#endif
