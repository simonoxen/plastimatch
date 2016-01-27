/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _bspline_score_h_
#define _bspline_score_h_

#include "plmregister_config.h"
#include <vector>
#include "plm_int.h"

class PLMREGISTER_API Bspline_score
{
public:
    Bspline_score ();
    ~Bspline_score ();
public:
    float score;           /* Total Score (sent to optimizer) */
    float lmetric;         /* Landmark metric */
    float rmetric;         /* Regularization metric */
    std::vector<float> smetric;  /* Similarity metric */

    plm_long num_vox;      /* Number of voxel with correspondence */
    plm_long num_coeff;    /* Size of gradient vector = num coefficents */
    float* smetric_grad;   /* Gradient of score for current smetric */
    float* total_grad;     /* Total cost function gradient wrt coefficients */

    /* Time to compute similarity metric */
    std::vector<double> time_smetric;
    /* Time to compute regularization metric */
    double time_rmetric;
public:
    void set_num_coeff (plm_long num_coeff);
    void reset_smetric_grad ();
    void reset_score ();
    void accumulate_grad (float lambda);
};

#endif
