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
    this->smetric = 0;
    this->num_vox = 0;

    this->num_coeff = 0;
    this->grad = 0;

    this->time_smetric = 0;
    this->time_rmetric = 0;
}

Bspline_score::~Bspline_score ()
{
    delete[] grad;
}

void
Bspline_score::set_num_coeff (plm_long num_coeff)
{
    this->num_coeff = num_coeff;
    delete[] this->grad;
    this->grad = new float[num_coeff];
}

void
Bspline_score::reset_score ()
{
    this->score = 0;
    this->lmetric = 0;
    this->rmetric = 0;
    this->smetric = 0;
    this->num_vox = 0;
    memset (this->grad, 0, this->num_coeff * sizeof(float));
    this->time_smetric = 0;
    this->time_rmetric = 0;
}
