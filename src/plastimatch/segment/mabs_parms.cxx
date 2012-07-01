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

#include "plmbase.h"
#include "plmsegment.h"
#include "plmsys.h"

#include "plm_math.h"

Mabs_parms::Mabs_parms ()
{
    this->atlas_dir[0] = '\0';
    this->training_dir[0] = '\0';
    this->registration_config[0] = '\0';
    this->sman = new Mabs_subject_manager;
    this->labeling_input_fn[0] = '\0';
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
    fprintf (stderr, "-- atlas_dir: %s\n", this->atlas_dir);
    fprintf (stderr, "-- training_dir: %s\n", this->training_dir);
    fprintf (stderr, "-- registration_config: %s\n", this->registration_config);
    while (sub) {
        fprintf (stderr, "-- subject\n");
        fprintf (stderr, "   -- img: %s [%p]\n", sub->img_fn, sub->img);
        fprintf (stderr, "   -- ss : %s [%p]\n", sub->ss_fn, sub->ss);
        sub = this->sman->next ();
    }
    fprintf (stderr, "-- labeling_input_fn: %s\n", this->labeling_input_fn);
    fprintf (stderr, "-- labeling_output_fn: %s\n", 
        this->labeling_output_fn.c_str());
}

int
Mabs_parms::set_key_val (
    const char* key, 
    const char* val, 
    int section
)
{
    Mabs_subject* subject = this->sman->current ();

    switch (section) {
    /* [TRAINING] */
    case 0:
        if (!strcmp (key, "atlas_dir")) {
            strncpy ((char*)this->atlas_dir, val, _MAX_PATH);
        }
        else if (!strcmp (key, "training_dir")) {
            strncpy ((char*)this->training_dir, val, _MAX_PATH);
        }
        break;

    /* [REGISTRATION] */
    case 1:
        if (!strcmp (key, "registration_config")) {
            strncpy ((char*)this->registration_config, val, _MAX_PATH);
        }
        break;

    /* [SUBJECT] */
    case 2:
        /* head is the most recent addition to the list */
        if (!strcmp (key, "image")) {
            strncpy ((char*)subject->img_fn, val, _MAX_PATH);
        }
        else if (!strcmp (key, "structs")) {
            strncpy ((char*)subject->ss_fn, val, _MAX_PATH);
        }
        break;

    /* [STRUCTURES] */
    case 3:
        // not yet implemented
        break;

    /* [LABELING] */
    case 4:
        if (!strcmp (key, "input")) {
            strncpy ((char*)this->labeling_input_fn, val, _MAX_PATH);
        }
        else if (!strcmp (key, "output")) {
            this->labeling_output_fn = val;
        }
    }
    return 0;

#if 0
  error_exit:
    print_and_exit ("Unknown (key,val) combination: (%s,%s)\n", key, val);
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
            if (buf.find ("[TRAINING]") != std::string::npos
                || buf.find ("[training]") != std::string::npos)
            {
                section = 0;
                continue;
            }
            else if (buf.find ("[REGISTRATION]") != std::string::npos
                || buf.find ("[registration]") != std::string::npos)
            {
                section = 1;
                continue;
            }
            else if (buf.find ("[SUBJECT]") != std::string::npos
                || buf.find ("[subject]") != std::string::npos)
            {
                section = 2;
                this->sman->add ();
                this->sman->select_head ();
                continue;
            }
            else if (buf.find ("[STRUCTURES]") != std::string::npos
                || buf.find ("[structures]") != std::string::npos)
            {
                section = 3;
                continue;
            }
            else if (buf.find ("[LABELING]") != std::string::npos
                || buf.find ("[labeling]") != std::string::npos)
            {
                section = 4;
                continue;
            }
            else {
                printf ("Parse error: %s\n", buf_ori.c_str());
            }
        }

        size_t key_loc = buf.find ("=");
        if (key_loc == std::string::npos) {
            continue;
        }

        std::string key = buf.substr (0, key_loc);
        std::string val = buf.substr (key_loc+1);
        key = trim (key);
        val = trim (val);

        if (key != "" && val != "") {
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
