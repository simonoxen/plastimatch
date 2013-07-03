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

class Mabs_subject_manager;

class PLMSEGMENT_API Mabs_parms {
public:
    Mabs_parms ();
    ~Mabs_parms ();

    bool parse_args (int argc, char** argv);
    void print ();

    void parse_config (const char* config_fn);
private:
    int set_key_val (const std::string& key, const std::string& val, 
        int section);

public:
    /* [PREALIGNMENT] */
    std::string prealignment_mode;
    
    /* [ATLASES-SELECTION] */
    bool enable_atlases_selection;
    float mi_percent_thershold;
    int mi_histogram_bins;
    std::string roi_mask_fn;
    int lower_mi_value;
    int upper_mi_value;
    
    /* [TRAINING] */
    std::string atlas_dir;
    std::string training_dir;

    std::string minsim_values;
    std::string rho_values;
    std::string sigma_values;
    std::string threshold_values;

    bool write_thresholded_files;
    bool write_weight_files;

    /* [REGISTRATION] */
    std::string registration_config;

    /* [SUBJECT] */
    Mabs_subject_manager* sman;    

    /* [STRUCTURES] */
    std::map<std::string, std::string> structure_map;

    /* [LABELING] */
    std::string labeling_input_fn;
    std::string labeling_output_fn;

    /* misc */
    bool debug;
};

#endif /* #ifndef _mabs_parms_h_ */
