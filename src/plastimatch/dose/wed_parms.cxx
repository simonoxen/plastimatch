/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plm_config.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <fstream>
#include <iostream>
#include <sstream>

#include "plm_math.h"
#include "print_and_exit.h"
#include "string_util.h"
#include "wed_parms.h"

#ifndef NULL
#define NULL ((void*)0)
#endif

Wed_Parms::Wed_Parms ()
{
    this->debug = 0;
    this->ray_step = 1.0f;
    this->input_ct_fn[0] = '\0';
    this->input_dose_fn[0] = '\0';
    this->output_ct_fn[0] = '\0';
    this->output_dose_fn[0] = '\0';

    this->src[0] = -1000.f;
    this->src[1] = 0.f;
    this->src[2] = 0.f;
    this->isocenter[0] = 0.f;
    this->isocenter[1] = 0.f;
    this->isocenter[2] = 0.f;
    this->beam_res = 1.f;

    this->vup[0] = 0.f;
    this->vup[1] = 0.f;
    this->vup[2] = 1.f;
    this->ires[0] = 200;
    this->ires[1] = 200;
    this->have_ic = false;
    this->ic[0] = 99.5f;
    this->ic[1] = 99.5f;
    this->ap_offset = 100;
}

Wed_Parms::~Wed_Parms ()
{
}

static void
print_usage (void)
{
    printf ("Usage: wed config_file\n");
    exit (1);
}

int
Wed_Parms::set_key_val (
    const char* key, 
    const char* val, 
    int section
)
{
    switch (section) {

        /* [SETTINGS] */
    case 0:
        if (!strcmp (key, "ray_step")) {
            if (sscanf (val, "%f", &this->ray_step) != 1) {
                goto error_exit;
            }
        }
        else if (!strcmp (key, "patient")) {
            strncpy (this->input_ct_fn, val, _MAX_PATH);
        }
        else if (!strcmp (key, "dose")) {
            strncpy (this->input_dose_fn, val, _MAX_PATH);
        }
        else if (!strcmp (key, "patient_wed")) {
            strncpy (this->output_ct_fn, val, _MAX_PATH);
        }
        else if (!strcmp (key, "dose_wed")) {
            strncpy (this->output_dose_fn, val, _MAX_PATH);
        }
        break;

        /* [BEAM] */
    case 1:
        if (!strcmp (key, "pos")) {
            if (sscanf (val, "%f %f %f", &(this->src[0]), &(this->src[1]), &(this->src[2])) != 3) {
                goto error_exit;
            }
        }
        else if (!strcmp (key, "isocenter")) {
            if (sscanf (val, "%f %f %f", &(this->isocenter[0]), &(this->isocenter[1]), &(this->isocenter[2])) != 3) {
                goto error_exit;
            }
        }
        else if (!strcmp (key, "res")) {
            if (sscanf (val, "%f", &(this->beam_res)) != 1) {
                goto error_exit;
            }
        }

        break;

        /* [APERTURE] */
    case 2:
        if (!strcmp (key, "up")) {
            if (sscanf (val, "%f %f %f", &(this->vup[0]), &(this->vup[1]), &(this->vup[2])) != 3) {
                goto error_exit;
            }
        }
        else if (!strcmp (key, "center")) {
            if (sscanf (val, "%f %f", &(this->ic[0]), &(this->ic[1])) != 2) {
                goto error_exit;
            }
            this->have_ic = true;
        }
        else if (!strcmp (key, "offset")) {
            if (sscanf (val, "%f", &(this->ap_offset)) != 1) {
                goto error_exit;
            }
        }
        else if (!strcmp (key, "resolution")) {
            if (sscanf (val, "%i %i", &(this->ires[0]), &(this->ires[1])) != 2) {
                goto error_exit;
            }
        }
        break;
    }
    return 0;

error_exit:
    print_and_exit ("Unknown (key,val) combination: (%s,%s)\n", key, val);
    return -1;
}



void
Wed_Parms::parse_config (
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
            if (buf.find ("[SETTINGS]") != std::string::npos
                || buf.find ("[settings]") != std::string::npos)
            {
                section = 0;
                continue;
            }
            else if (buf.find ("[BEAM]") != std::string::npos
                || buf.find ("[beam]") != std::string::npos)
            {
                section = 1;
                continue;
            }
            else if (buf.find ("[APERTURE]") != std::string::npos
                || buf.find ("[aperture]") != std::string::npos)
            {
                section = 2;
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
Wed_Parms::parse_args (int argc, char** argv)
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

#if 0
    if (this->d_lut == NULL) {
        /* measured bragg curve not supplied, try to generate */
        if (!this->generate ()) {
            return false;
        }
    }
    // JAS 2012.08.10
    //   Hack so that I can reuse the proton code.  The values
    //   don't actually matter.
    scene->beam->E0 = 1.0;
    scene->beam->spread = 1.0;
    scene->beam->dmax = 1.0;
#endif

    if (this->output_ct_fn[0] == '\0') {
        fprintf (stderr, "\n** ERROR: Output file for patient water equivalent depth volume not specified in configuration file!\n");
        return false;
    }

#if 0
    if (this->output_dose_fn[0] == '\0') {
        fprintf (stderr, "\n** ERROR: Output file for dose water equivalent depth volume not specified in configuration file!\n");
        return false;
    }
#endif

    if (this->input_ct_fn[0] == '\0') {
        fprintf (stderr, "\n** ERROR: Input patient image not specified in configuration file!\n");
        return false;
    }

    return true;
}
