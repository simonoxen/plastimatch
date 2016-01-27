/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plmregister_config.h"
#include <string.h>

#include "bspline_score.h"

Bspline_score::Bspline_score ()
{
    this->score = 0;
    this->lmetric = 0;
    this->rmetric = 0;

    this->num_vox = 0;
    this->num_coeff = 0;
    this->smetric_grad = 0;
    this->total_grad = 0;

    this->time_rmetric = 0;
}

Bspline_score::~Bspline_score ()
{
    delete[] smetric_grad;
    delete[] total_grad;
}

void
Bspline_score::set_num_coeff (plm_long num_coeff)
{
    this->num_coeff = num_coeff;
    delete[] this->smetric_grad;
    delete[] this->total_grad;
    this->smetric_grad = new float[num_coeff];
    this->total_grad = new float[num_coeff];
}

void
Bspline_score::reset_smetric_grad ()
{
    memset (this->smetric_grad, 0, this->num_coeff * sizeof(float));
}

void
Bspline_score::reset_score ()
{
    this->score = 0;
    this->lmetric = 0;
    this->rmetric = 0;
    this->smetric.clear();
    this->num_vox = 0;
    memset (this->smetric_grad, 0, this->num_coeff * sizeof(float));
    memset (this->total_grad, 0, this->num_coeff * sizeof(float));
    this->time_smetric.clear();
    this->time_rmetric = 0;
}

void
Bspline_score::accumulate_grad (float lambda)
{
    for (plm_long i = 0; i < this->num_coeff; i++) {
        this->total_grad[i] += lambda * this->smetric_grad[i];
    }
    this->reset_smetric_grad ();
}
