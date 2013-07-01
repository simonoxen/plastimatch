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

#include "mabs_parms.h"
#include "mabs_subject.h"
#include "plm_math.h"
#include "string_util.h"

Mabs_parms::Mabs_parms ()
{
    /* [PREALIGNMENT] */
    this->prealignment_mode="default";
    
    /* [ATLASES-SELECTION] */
    this->enable_atlases_selection = false;
    this->mi_percent_thershold = 0.70;
    this->roi_mask_fn = "";
    this->lower_mi_value = 0;
    this->upper_mi_value = 0;
	
    this->sman = new Mabs_subject_manager;
    this->debug = false;
    this->minsim_values = "L 0.0001:1:0.0001";
    this->rho_values = "1:1:1";
    this->sigma_values = "L 1.7:1:1.7";
    this->threshold_values = "0.5";
    this->write_thresholded_files = true;
    this->write_weight_files = true;
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
    int section
)
{
    Mabs_subject* subject = this->sman->current ();

    switch (section) {
    /* [PREALIGNMENT] */
    case 0:
    if (key == "prealignment_mode") {
        if (val == "DISABLE" || val == "disable" || val == "Disable" || val == "0") {
            this->prealignment_mode = "disable";
        }
        else if (val == "DEFAULT" || val == "default" || val == "Default") {
            this->prealignment_mode = "default";
        }
       else if (val == "CUSTOM" || val == "custom" || val == "Custom") {
            this->prealignment_mode = "custom";
        }
    }
    break;

    /* [ATLASES-SELECTION] */
    case 1:
    if (key == "enable_atlases_selection") {
        if (val == "True" || val == "true" || val == "1") {
            this->enable_atlases_selection = true;
        }
        else {
            this->enable_atlases_selection = false;
        }   
    }
    else if (key == "mi_percent_thershold") {
        sscanf (val.c_str(), "%g", &this->mi_percent_thershold);
    }
    else if (key == "roi_mask_fn" || key == "roi_mask") {
        this->roi_mask_fn = val;
    }
    else if (key == "lower_mi_value") {
        sscanf (val.c_str(), "%d", &this->lower_mi_value);
    }
    else if (key == "upper_mi_value") {
        sscanf (val.c_str(), "%d", &this->upper_mi_value);
    }
    
    break;
        	
    /* [TRAINING] */
    case 2:
        if (key == "atlas_dir") {
            this->atlas_dir = val;
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
        break;

    /* [REGISTRATION] */
    case 3:
        if (key == "registration_config") {
            this->registration_config = val;
        }
        break;

    /* [SUBJECT] */
    case 4:
        /* head is the most recent addition to the list */
        if (key == "image") {
            strncpy ((char*)subject->img_fn, val.c_str(), _MAX_PATH);
        }
        else if (key == "structs") {
            strncpy ((char*)subject->ss_fn, val.c_str(), _MAX_PATH);
        }
        break;

    /* [STRUCTURES] */
    case 5:
        /* Add key to list of structures */
        this->structure_map[key] = key;
        if (val != "") {
            /* Key/value pair, so add for renaming */
            this->structure_map[val] = key;
        }
        break;

    /* [LABELING] */
    case 6:
        if (key == "input") {
            this->labeling_input_fn = val;
        }
        else if (key == "output") {
            this->labeling_output_fn = val;
        }
    }
    return 0;

#if 0
  error_exit:
    print_and_exit ("Unknown (key,val) combination: (%s,%s)\n", 
        key.c_str(), val.c_str());
    return -1;
#endif
}

void
Mabs_parms::parse_config (
    const char* config_fn
)
{
    /* Read file into string */
    std::ifstream t (config_fn);
    std::stringstream buffer;
    buffer << t.rdbuf();

    std::string buf;
    std::string buf_ori;    /* An extra copy for diagnostics */
    int section = 0;

    std::stringstream ss (buffer.str());

    while (getline (ss, buf)) {
        buf_ori = buf;
        buf = trim (buf);
        buf_ori = trim (buf_ori, "\r\n");

        if (buf == "") continue;
        if (buf[0] == '#') continue;

        if (buf[0] == '[') {
            if (buf.find ("[ATLASES-SELECTION]") != std::string::npos
                || buf.find ("[atlases-selection]") != std::string::npos)
            {
                section = 1;
                continue;
            }
            else if (buf.find ("[TRAINING]") != std::string::npos
                || buf.find ("[training]") != std::string::npos)
            {
                section = 2;
                continue;
            }
            else if (buf.find ("[REGISTRATION]") != std::string::npos
                || buf.find ("[registration]") != std::string::npos)
            {
                section = 3;
                continue;
            }
            else if (buf.find ("[SUBJECT]") != std::string::npos
                || buf.find ("[subject]") != std::string::npos)
            {
                section = 4;
                this->sman->add ();
                this->sman->select_head ();
                continue;
            }
            else if (buf.find ("[STRUCTURES]") != std::string::npos
                || buf.find ("[structures]") != std::string::npos)
            {
                section = 5;
                continue;
            }
            else if (buf.find ("[LABELING]") != std::string::npos
                || buf.find ("[labeling]") != std::string::npos)
            {
                section = 6;
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
            if (this->set_key_val (key.c_str(), val.c_str(), section) < 0) {
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
