/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _bspline_score_h_
#define _bspline_score_h_

#include "plmregister_config.h"
#include "plm_int.h"

class PLMREGISTER_API Bspline_score {
public:
    float score;         /* Total Score (sent to optimizer) */
    float lmetric;       /* Landmark metric */
    float rmetric;       /* Regularization metric */
    float smetric;       /* Similarity metric */
    plm_long num_vox;    /* Number of voxel with correspondence */

    float* grad;         /* Gradient score wrt control coeff */

    double time_smetric;   /* Time to compute similarity metric */
    double time_rmetric;   /* Time to compute regularization metric */
public:
    Bspline_score ();
};

#endif
