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
#include "mabs_parms.h"
#include "mabs_subject.h"
#include "plm_math.h"
#include "print_and_exit.h"
#include "string_util.h"

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
    this->mi_percent_threshold = 0.70;
    this->mi_histogram_bins = 100;
    this->roi_mask_fn = "";
    this->nmi_ratio_registration_config_fn = "";
    this->lower_mi_value_sub_defined=false;
    this->lower_mi_value_sub = 0;
    this->upper_mi_value_sub_defined=false;
    this->upper_mi_value_sub = 0;
    this->lower_mi_value_atl_defined=false;
    this->lower_mi_value_atl = 0;
    this->upper_mi_value_atl_defined=false;
    this->upper_mi_value_atl = 0;
    this->min_random_atlases = 4;
    this->max_random_atlases = 12;
    this->precomputed_ranking_fn = "";
    this->atlases_from_precomputed_ranking = 0;
    this->atlases_from_precomputed_ranking_defined = false;
	
    /* [TRAINING] */
    this->distance_map_algorithm = "";

    this->minsim_values = "L 0.0001:1:0.0001";
    this->rho_values = "1:1:1";
    this->sigma_values = "L 1.7:1:1.7";
    this->threshold_values = "0.5";

    this->write_thresholded_files = true;
    this->write_weight_files = true;
    this->write_warped_images = true;
    this->write_warped_structures = true;

    /* [SUBJECT] */
    this->sman = new Mabs_subject_manager;

    /* misc */
    this->debug = false;
}

Mabs_parms::~Mabs_parms ()
{
    delete this->sman;
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

void
Mabs_parms::print ()
{
    Mabs_subject* sub = this->sman->current ();

    fprintf (stderr, "Mabs_parms:\n");
    fprintf (stderr, "-- atlas_dir: %s\n", this->atlas_dir.c_str());
    fprintf (stderr, "-- training_dir: %s\n", this->training_dir.c_str());
    fprintf (stderr, "-- registration_config: %s\n", 
        this->registration_config.c_str());
    while (sub) {
        fprintf (stderr, "-- subject\n");
        fprintf (stderr, "   -- img: %s [%p]\n", sub->img_fn, sub->img);
        fprintf (stderr, "   -- ss : %s [%p]\n", sub->ss_fn, sub->ss);
        sub = this->sman->next ();
    }
    fprintf (stderr, "-- labeling_input_fn: %s\n", 
        this->labeling_input_fn.c_str());
    fprintf (stderr, "-- labeling_output_fn: %s\n", 
        this->labeling_output_fn.c_str());
}

int
Mabs_parms::set_key_val (
    const std::string& key, 
    const std::string& val, 
    const std::string& section
)
{
    Mabs_subject* subject = this->sman->current ();

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
            else if (val == "nmi-ratio" || val == "NMI-RATIO") {
                this->atlas_selection_criteria="nmi-ratio";
            }
            else if (val == "random" || val == "RANDOM") { // Just for testing purpose
                this->atlas_selection_criteria="random";
            }
            else if (val == "precomputed" || val == "PRECOMPUTED") { // Just for testing purpose
                this->atlas_selection_criteria="precomputed";
            }
       }
        else if (key == "mi_percent_threshold") {
            sscanf (val.c_str(), "%g", &this->mi_percent_threshold);
        }
        else if (key == "mi_histogram_bins") {
            sscanf (val.c_str(), "%d", &this->mi_histogram_bins);
        }
        else if (key == "roi_mask_fn" || key == "roi_mask") {
            this->roi_mask_fn = val;
        }
         else if (key == "nmi_ratio_registration_config") {
            this->nmi_ratio_registration_config_fn = val;
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
         else if (key == "atlases_from_precomputed_ranking") {
            sscanf (val.c_str(), "%d", &this->atlases_from_precomputed_ranking);
            this->atlases_from_precomputed_ranking_defined = true;
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

    /* [SUBJECT] */
    if (section == "SUBJECT") {
        /* head is the most recent addition to the list */
        if (key == "image") {
            strncpy ((char*)subject->img_fn, val.c_str(), _MAX_PATH);
        }
        else if (key == "structs") {
            strncpy ((char*)subject->ss_fn, val.c_str(), _MAX_PATH);
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
    /* Confirm file can be read */
    if (!file_exists (config_fn)) {
        print_and_exit ("Error reading config file: %s\n", config_fn);
    }

    /* Read file into string */
    std::ifstream t (config_fn);
    std::stringstream buffer;
    buffer << t.rdbuf();

    std::string buf;
    std::string buf_ori;    /* An extra copy for diagnostics */
    std::string section = "";

    std::stringstream ss (buffer.str());

    while (getline (ss, buf)) {
        buf_ori = buf;
        buf = trim (buf);
        buf_ori = trim (buf_ori, "\r\n");

        if (buf == "") continue;
        if (buf[0] == '#') continue;

        if (buf[0] == '[') {
            if (buf.find ("[PREALIGNMENT]") != std::string::npos
                || buf.find ("[prealignment]") != std::string::npos)
            {
                section = "PREALIGNMENT";
                continue;
            }
            else if (buf.find ("[ATLAS-SELECTION]") != std::string::npos
                || buf.find ("[atlas-selection]") != std::string::npos)
            {
                section = "ATLAS-SELECTION";
                continue;
            }
            else if (buf.find ("[TRAINING]") != std::string::npos
                || buf.find ("[training]") != std::string::npos)
            {
                section = "TRAINING";
                continue;
            }
            else if (buf.find ("[REGISTRATION]") != std::string::npos
                || buf.find ("[registration]") != std::string::npos)
            {
                section = "REGISTRATION";
                continue;
            }
            else if (buf.find ("[SUBJECT]") != std::string::npos
                || buf.find ("[subject]") != std::string::npos)
            {
                section = "SUBJECT";
                this->sman->add ();
                this->sman->select_head ();
                continue;
            }
            else if (buf.find ("[STRUCTURES]") != std::string::npos
                || buf.find ("[structures]") != std::string::npos)
            {
                section = "STRUCTURES";
                continue;
            }
            else if (buf.find ("[LABELING]") != std::string::npos
                || buf.find ("[labeling]") != std::string::npos)
            {
                section = "LABELING";
                continue;
            }
            else {
                printf ("Parse error: %s\n", buf_ori.c_str());
            }
        }

        std::string key;
        std::string val;
        size_t key_loc = buf.find ("=");
        if (key_loc == std::string::npos) {
            key = buf;
            val = "";
        } else {
            key = buf.substr (0, key_loc);
            val = buf.substr (key_loc+1);
        }
        key = trim (key);
        val = trim (val);

        if (key != "") {
            if (this->set_key_val (key, val, section) < 0) {
                printf ("Parse error: %s\n", buf_ori.c_str());
            }
        }
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
