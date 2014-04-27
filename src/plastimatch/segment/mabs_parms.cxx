/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plmsegment_config.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <fstream>
#include <iostream>
#include <sstream>

#include "file_util.h"
#include "parameter_parser.h"
#include "mabs_parms.h"
#include "plm_math.h"
#include "print_and_exit.h"
#include "string_util.h"

class Mabs_parms_parser : public Parameter_parser
{
public:
    Mabs_parms *mp;
public:
    Mabs_parms_parser (Mabs_parms *mp)
    {
        this->mp = mp;
    }
public:
    virtual int process_section (
        const std::string& section)
    {
        if (section == "PREALIGN" || section == "PREALIGNMENT") {
            this->enable_key_regularization (true);
            return 0;
        }
        if (section == "ATLAS-SELECTION") {
            this->enable_key_regularization (true);
            return 0;
        }
        if (section == "TRAINING") {
            this->enable_key_regularization (true);
            return 0;
        }
        if (section == "REGISTRATION") {
            this->enable_key_regularization (true);
            return 0;
        }
        if (section == "STRUCTURES") {
            this->enable_key_regularization (false);
            return 0;
        }
        if (section == "LABELING") {
            this->enable_key_regularization (true);
            return 0;
        }
        if (section == "OPTIMIZATION_RESULT") {
            this->enable_key_regularization (true);
            return 0;
        }

        /* else, unknown section */
        return -1;
    }
    virtual int process_key_value (
        const std::string& section,
        const std::string& key, 
        const std::string& val)
    {
        return this->mp->set_key_value (section, key, val);
    }
};

Mabs_parms::Mabs_parms ()
{
    /* [PREALIGNMENT] */
    this->prealign_mode="default";
    this->prealign_reference = "";
    this->prealign_spacing = "";
    this->prealign_registration_config = "";

    /* [ATLAS-SELECTION] */
    this->enable_atlas_selection = false;
    this->atlas_selection_criteria="nmi";
    this->similarity_percent_threshold = 0.40;
    this->atlases_from_ranking = -1;
    this->mi_histogram_bins = 100;
    this->roi_mask_fn = "";
    this->selection_reg_parms_fn = "";
    this->lower_mi_value_sub_defined=false;
    this->lower_mi_value_sub = 0;
    this->upper_mi_value_sub_defined=false;
    this->upper_mi_value_sub = 0;
    this->lower_mi_value_atl_defined=false;
    this->lower_mi_value_atl = 0;
    this->upper_mi_value_atl_defined=false;
    this->upper_mi_value_atl = 0;
    this->max_random_atlases = 14;
    this->min_random_atlases = 6;
    this->precomputed_ranking_fn = "";
	
    /* [TRAINING] */
    this->distance_map_algorithm = "";

    this->minsim_values = "L 0.0001:1:0.0001";
    this->rho_values = "1:1:1";
    this->sigma_values = "L 1.7:1:1.7";
    this->threshold_values = "0.5";

    this->write_distance_map_files = true;
    this->write_thresholded_files = true;
    this->write_weight_files = true;
    this->write_warped_images = true;
    this->write_warped_structures = true;

    /* [OPTIMIZATION-RESULT] */
    this->optimization_result_reg = "";
    this->optimization_result_seg_rho = 1.0;
    this->optimization_result_seg_sigma = 30.;
    this->optimization_result_seg_minsim = 0.3;
    this->optimization_result_seg_thresh = "0.4";

    /* misc */
    this->debug = false;
}

Mabs_parms::~Mabs_parms ()
{
}

static void
print_usage ()
{
    printf (
        "Usage: mabs [options] config_file\n"
        "Options:\n"
        " --debug           Enable various debug output\n"
    );
    exit (1);
}

int
Mabs_parms::set_key_value (
    const std::string& section, 
    const std::string& key, 
    const std::string& val
)
{
    /* [PREALIGNMENT] */
    if (section == "PREALIGN" || section == "PREALIGNMENT") {
        if (key == "mode") {
            if (val == "DISABLE" || val == "disable" 
                || val == "Disable" || val == "0")
            {
                this->prealign_mode = "disable";
            }
            else if (val == "DEFAULT" || val == "default" || val == "Default") {
                this->prealign_mode = "default";
            }
            else if (val == "CUSTOM" || val == "custom" || val == "Custom") {
                this->prealign_mode = "custom";
            }
        }
        else if (key == "reference") {
            this->prealign_reference = val;
        }
        else if (key == "spacing") {
            this->prealign_spacing = val;
        }
        else if (key == "registration_config") {
            this->prealign_registration_config = val;
        }
        else {
            goto error_exit;
        }
    }

    /* [ATLAS-SELECTION] */
    if (section == "ATLAS-SELECTION") {
        if (key == "enable_atlas_selection") {
            if (val == "True" || val == "true" || val == "1") {
                this->enable_atlas_selection = true;
            }
            else {
                this->enable_atlas_selection = false;
            }   
        }
        else if (key == "atlas_selection_criteria") {
            if (val == "nmi" || val == "NMI") {
                this->atlas_selection_criteria="nmi";
            }
            else if (val == "nmi-post" || val == "NMI-POST") {
                this->atlas_selection_criteria="nmi-post";
            }
            else if (val == "nmi-ratio" || val == "NMI-RATIO") {
                this->atlas_selection_criteria="nmi-ratio";
            }
            else if (val == "mse" || val == "MSE") {
                this->atlas_selection_criteria="mse";
            }
            else if (val == "mse-post" || val == "MSE-POST") {
                this->atlas_selection_criteria="mse-post";
            }
            else if (val == "mse-ratio" || val == "MSE-RATIO") {
                this->atlas_selection_criteria="mse-ratio";
            }
            else if (val == "random" || val == "RANDOM") {
                this->atlas_selection_criteria="random";
            }
            else if (val == "precomputed" || val == "PRECOMPUTED") {
                this->atlas_selection_criteria="precomputed";
            }
        }
        else if (key == "similarity_percent_threshold") {
            sscanf (val.c_str(), "%g", &this->similarity_percent_threshold);
        }
        else if (key == "atlases_from_ranking") {
            sscanf (val.c_str(), "%d", &this->atlases_from_ranking);
        }
        else if (key == "mi_histogram_bins") {
            sscanf (val.c_str(), "%d", &this->mi_histogram_bins);
        }
        else if (key == "roi_mask_fn" || key == "roi_mask") {
            this->roi_mask_fn = val;
        }
        else if (key == "selection_reg_parms") {
            this->selection_reg_parms_fn = val;
        }
        else if (key == "lower_mi_value_subject") {
            sscanf (val.c_str(), "%d", &this->lower_mi_value_sub);
            this->lower_mi_value_sub_defined = true;
        }
        else if (key == "upper_mi_value_subject") {
            sscanf (val.c_str(), "%d", &this->upper_mi_value_sub);
            this->upper_mi_value_sub_defined = true;
        }
        else if (key == "lower_mi_value_atlas") {
            sscanf (val.c_str(), "%d", &this->lower_mi_value_atl);
            this->lower_mi_value_atl_defined = true;
        }
        else if (key == "upper_mi_value_atlas") {
            sscanf (val.c_str(), "%d", &this->upper_mi_value_atl);
            this->upper_mi_value_atl_defined = true;
        }
        else if (key == "min_random_atlases") {
            sscanf (val.c_str(), "%d", &this->min_random_atlases);
        }
        else if (key == "max_random_atlases") {
            sscanf (val.c_str(), "%d", &this->max_random_atlases);
        }
        else if (key == "precomputed_ranking") {
            this->precomputed_ranking_fn = val;
        }
        else {
            goto error_exit;
        }
    }
        	
    /* [TRAINING] */
    if (section == "TRAINING") {
        if (key == "atlas_dir") {
            this->atlas_dir = val;
        }
        else if (key == "distance_map_algorithm") {
            this->distance_map_algorithm = val;
        }
        else if (key == "minimum_similarity") {
            this->minsim_values = val;
        }
        else if (key == "rho_values") {
            this->rho_values = val;
        }
        else if (key == "sigma_values") {
            this->sigma_values = val;
        }
        else if (key == "threshold_values") {
            this->threshold_values = val;
        }
        else if (key == "training_dir") {
            this->training_dir = val;
        }
        else if (key == "write_distance_map_files") {
            if (val == "0") {
                this->write_distance_map_files = false;
            }
        }
        else if (key == "write_thresholded_files") {
            if (val == "0") {
                this->write_thresholded_files = false;
            }
        }
        else if (key == "write_weight_files") {
            if (val == "0") {
                this->write_weight_files = false;
            }
        }
        else if (key == "write_warped_images") {
            if (val == "0") {
                this->write_warped_images = false;
            }
        }
        else if (key == "write_warped_structures") {
            if (val == "0") {
                this->write_warped_structures = false;
            }
        }
        else {
            goto error_exit;
        }
    }

    /* [REGISTRATION] */
    if (section == "REGISTRATION") {
        if (key == "registration_config") {
            this->registration_config = val;
        }
        else {
            goto error_exit;
        }
    }

    /* [STRUCTURES] */
    if (section == "STRUCTURES") {
        /* Add key to list of structures */
        this->structure_map[key] = key;
        if (val != "") {
            /* Key/value pair, so add for renaming */
            this->structure_map[val] = key;
        }
        /* There is no filtering of structures values */
    }


    /* [LABELING] */
    if (section == "LABELING") {
        if (key == "input") {
            this->labeling_input_fn = val;
        }
        else if (key == "output") {
            this->labeling_output_fn = val;
        }
        else {
            goto error_exit;
        }
    }

    /* [OPTIMIZATION-RESULT] */
    if (section == "OPTIMIZATION_RESULT") {
        if (key == "registration") {
            this->optimization_result_reg = val;
        }
        else if (key == "gaussian_weighting_voting_rho") {
            sscanf (val.c_str(), "%g", &this->optimization_result_seg_rho);
        }
        else if (key == "gaussian_weighting_voting_sigma") {
            sscanf (val.c_str(), "%g", &this->optimization_result_seg_sigma);
        }
        else if (key == "gaussian_weighting_voting_minsim") {
            sscanf (val.c_str(), "%g", &this->optimization_result_seg_minsim);
        }
        else if (key == "gaussian_weighting_voting_thresh") {
            this->optimization_result_seg_thresh = val;
        }
        else {
            goto error_exit;
        }
    }

    return 0;

error_exit:
    print_and_exit ("Unknown (sec,key,val) combination: (%s,%s,%s)\n", 
        section.c_str(), key.c_str(), val.c_str());
    return -1;
}

void
Mabs_parms::parse_config (
    const char* config_fn
)
{
    Mabs_parms_parser mpp (this);

    /* Parse the main config file */
    mpp.parse_config_file (config_fn);

    /* After parsing main config file, also parse 
       optimization result files */

    std::string reg_fn = string_format (
        "%s/mabs-train/optimization_result_reg.txt",
        this->training_dir.c_str());
    std::string seg_fn = string_format (
        "%s/mabs-train/optimization_result_seg.txt",
        this->training_dir.c_str());
    if (file_exists (reg_fn)) {
        mpp.parse_config_file (reg_fn.c_str());
    }
    if (file_exists (seg_fn)) {
        mpp.parse_config_file (seg_fn.c_str());
    }
}

bool
Mabs_parms::parse_args (int argc, char** argv)
{
    int i;
    for (i=1; i<argc; i++) {
        if (argv[i][0] != '-') break;

        if (!strcmp (argv[i], "--debug")) {
            this->debug = 1;
        }
        else {
            print_usage ();
            break;
        }
    }

    if (!argv[i]) {
        print_usage ();
    } else {
        this->parse_config (argv[i]);
    }

    return true;
}
