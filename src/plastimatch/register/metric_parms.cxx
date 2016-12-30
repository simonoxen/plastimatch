/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plmregister_config.h"
#include <string>
#include "metric_parms.h"
#include "string_util.h"

Metric_parms::Metric_parms ()
{
    metric_type.push_back (SIMILARITY_METRIC_MSE);
    metric_lambda.push_back (1.0);
}

void
Metric_parms::set_metric_type (const std::string& val)
{
    this->metric_type.clear();
    std::vector<std::string> metric_vec = string_split (val, ',');
    if (metric_vec.size() == 0) {
        return;
    }

    for (size_t i = 0; i < metric_vec.size(); i++) {
        if (metric_vec[i] == "gm") {
            this->metric_type.push_back (SIMILARITY_METRIC_GM);
        }
        else if (metric_vec[i] == "mattes") {
            this->metric_type.push_back (SIMILARITY_METRIC_MI_MATTES);
        }
        else if (metric_vec[i] == "mse" || metric_vec[i] == "MSE") {
            this->metric_type.push_back (SIMILARITY_METRIC_MSE);
        }
        else if (metric_vec[i] == "mi" || metric_vec[i] == "MI") {
#if PLM_CONFIG_LEGACY_MI_METRIC
            this->metric_type.push_back (SIMILARITY_METRIC_MI_VW);
#else
            this->metric_type.push_back (SIMILARITY_METRIC_MI_MATTES);
#endif
        }
        else if (metric_vec[i] == "mi_vw"
            || metric_vec[i] == "viola-wells")
        {
            this->metric_type.push_back (SIMILARITY_METRIC_MI_VW);
        }
        else if (metric_vec[i] == "nmi" || metric_vec[i] == "NMI") {
            this->metric_type.push_back (SIMILARITY_METRIC_NMI);
        }
        else {
            this->metric_type.clear();
            return;
        }
    }
}
