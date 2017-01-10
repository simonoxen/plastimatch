/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _stage_similarity_data_h_
#define _stage_similarity_data_h_

#include "plmregister_config.h"
#include "similarity_metric_type.h"
#include "volume.h"

class PLMREGISTER_API Stage_similarity_data
{
public:
    SMART_POINTER_SUPPORT (Stage_similarity_data);
public:
    Stage_similarity_data () {
        metric_type = SIMILARITY_METRIC_MSE;
        metric_lambda = 1.f;
    }
public:
    Volume::Pointer fixed_ss;
    Volume::Pointer moving_ss;
    Volume::Pointer fixed_grad;
    Volume::Pointer moving_grad;
    Volume::Pointer fixed_roi;
    Volume::Pointer moving_roi;

    Similarity_metric_type metric_type;
    float metric_lambda;
public:
    const char *metric_string () {
        return similarity_metric_type_string (metric_type);
    }
};

#endif
