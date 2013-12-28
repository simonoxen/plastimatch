/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _mabs_parms_h_
#define _mabs_parms_h_

#include "plmsegment_config.h"
#include <list>
#include <map>
#include <string>
#include "plm_path.h"
#include "pstring.h"

class PLMSEGMENT_API Mabs_parms {
public:
    Mabs_parms ();
    ~Mabs_parms ();

public:
    bool parse_args (int argc, char** argv);
    void parse_config (const char* config_fn);
    int set_key_value (
        const std::string& section,
        const std::string& key, 
        const std::string& val);

public:
    /* [PREALIGNMENT] */
    std::string prealign_mode;
    std::string prealign_reference;
    std::string prealign_spacing;
    std::string prealign_registration_config;
    
    /* [ATLASES-SELECTION] */
    bool enable_atlas_selection;
    std::string atlas_selection_criteria;
    float mi_percent_threshold;
    int atlases_from_ranking;
    bool atlases_from_ranking_defined;
    int mi_histogram_bins;
    std::string roi_mask_fn;
    std::string selection_reg_parms_fn;
    bool lower_mi_value_sub_defined;
    int lower_mi_value_sub;
    bool upper_mi_value_sub_defined;
    int upper_mi_value_sub;
    bool lower_mi_value_atl_defined;
    int lower_mi_value_atl;
    bool upper_mi_value_atl_defined;
    int upper_mi_value_atl;
    int min_random_atlases;
    int max_random_atlases;
    std::string precomputed_ranking_fn;
    
    /* [TRAINING] */
    std::string atlas_dir;
    std::string training_dir;

    std::string distance_map_algorithm;
    std::string minsim_values;
    std::string rho_values;
    std::string sigma_values;
    std::string threshold_values;

    bool write_distance_map_files;
    bool write_thresholded_files;
    bool write_weight_files;
    bool write_warped_images;
    bool write_warped_structures;

    /* [REGISTRATION] */
    std::string registration_config;

    /* [STRUCTURES] */
    std::map<std::string, std::string> structure_map;

    /* [LABELING] */
    std::string labeling_input_fn;
    std::string labeling_output_fn;

    /* [OPTIMIZATION-RESULT] */
    std::string optimization_result_reg;

    /* misc */
    bool debug;
};

#endif /* #ifndef _mabs_parms_h_ */
