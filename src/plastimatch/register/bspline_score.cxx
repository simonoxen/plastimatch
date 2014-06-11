/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plmregister_config.h"

#include "bspline_score.h"

Bspline_score::Bspline_score ()
{
    this->score = 0;
    this->lmetric = 0;
    this->rmetric = 0;
    this->smetric = 0;
    this->num_vox = 0;
    this->grad = 0;

    this->time_smetric = 0;
    this->time_rmetric = 0;
}
