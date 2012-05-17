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

#include "plmdose.h"
#include "plmsys.h"

#include "plm_math.h"

#ifndef NULL
#define NULL ((void*)0)
#endif

void
print_usage (void)
{
    printf (
        "Usage: proton_dose [options] config_file\n"
        "Options:\n"
        " --debug           Create various debug files\n"
    );
    exit (1);
}

int
Proton_dose_parms::set_key_val (
    const char* key, 
    const char* val, 
    int section
)
{
    switch (section) {

    /* [SETTINGS] */
    case 0:
        if (!strcmp (key, "flavor")) {
            if (strlen (val) >= 1) {
                this->flavor = val[0];
            } else {
                goto error_exit;
            } 
        }
        else if (!strcmp (key, "threading")) {
            if (!strcmp (val,"single")) {
                this->threading = THREADING_CPU_SINGLE;
            }
            else if (!strcmp (val,"openmp")) {
#if (OPENMP_FOUND)
                this->threading = THREADING_CPU_OPENMP;
#else
                this->threading = THREADING_CPU_SINGLE;
#endif
            }
            else if (!strcmp (val,"cuda")) {
#if (CUDA_FOUND)
                this->threading = THREADING_CUDA;
#elif (OPENMP_FOUND)
                this->threading = THREADING_CPU_OPENMP;
#else
                this->threading = THREADING_CPU_SINGLE;
#endif
            }
            else {
                goto error_exit;
            }
        }
        else if (!strcmp (key, "ray_step")) {
            if (sscanf (val, "%f", &this->ray_step) != 1) {
                goto error_exit;
            }
        }
        else if (!strcmp (key, "scale")) {
            if (sscanf (val, "%f", &this->scale) != 1) {
                goto error_exit;
            }
        }
        else if (!strcmp (key, "detail")) {
            if (!strcmp (val, "low")) {
                this->detail = 1;
            }
            else if (!strcmp (val, "high")) {
                this->detail = 0;
            }
            else {
                goto error_exit;
            }
        }
        else if (!strcmp (key, "patient")) {
            strncpy (this->input_fn, val, _MAX_PATH);
        }
        else if (!strcmp (key, "dose")) {
            strncpy (this->output_fn, val, _MAX_PATH);
        }

        break;

    /* [BEAM] */
    case 1:
        if (!strcmp (key, "bragg_curve")) {
            this->beam->load (val);
        }
        else if (!strcmp (key, "pos")) {
            if (sscanf (val, "%lf %lf %lf", &(this->beam->src[0]), &(this->beam->src[1]), &(this->beam->src[2])) != 3) {
                goto error_exit;
            }
        }
        else if (!strcmp (key, "isocenter")) {
            if (sscanf (val, "%lf %lf %lf", &(this->beam->isocenter[0]), &(this->beam->isocenter[1]), &(this->beam->isocenter[2])) != 3) {
                goto error_exit;
            }
        }
        else if (!strcmp (key, "energy")) {
            if (sscanf (val, "%lf", &(this->beam->E0)) != 1) {
                goto error_exit;
            }
        }
        else if (!strcmp (key, "spread")) {
            if (sscanf (val, "%lf", &(this->beam->spread)) != 1) {
                goto error_exit;
            }
        }
        else if (!strcmp (key, "depth")) {
            if (sscanf (val, "%lf", &(this->beam->dmax)) != 1) {
                goto error_exit;
            }
        }
        else if (!strcmp (key, "res")) {
            if (sscanf (val, "%lf", &(this->beam->dres)) != 1) {
                goto error_exit;
            }
        }

        break;

    /* [APERTURE] */
    case 2:
        if (!strcmp (key, "up")) {
            if (sscanf (val, "%lf %lf %lf", &(this->aperture->vup[0]), &(this->aperture->vup[1]), &(this->aperture->vup[2])) != 3) {
                goto error_exit;
            }
        }
        else if (!strcmp (key, "center")) {
            if (sscanf (val, "%lf %lf", &(this->aperture->ic[0]), &(this->aperture->ic[1])) != 2) {
                goto error_exit;
            }
        }
        else if (!strcmp (key, "offset")) {
            if (sscanf (val, "%lf", &(this->aperture->ap_offset)) != 1) {
                goto error_exit;
            }
        }
        else if (!strcmp (key, "resolution")) {
            if (sscanf (val, "%i %i", &(this->aperture->ires[0]), &(this->aperture->ires[1])) != 2) {
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
Proton_dose_parms::parse_config (
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

void
Proton_dose_parms::parse_args (int argc, char** argv)
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

    if (this->beam->d_lut == NULL) {
        /* measured bragg curve not supplied, try to generate */
        if (!this->beam->generate ()) {
            exit (0);
        }
    }

    if (this->input_fn[0] == '\0') {
        fprintf (stderr, "\n** ERROR: Patient image not specified in configuration file!\n");
        exit (0);
    }
    if (this->output_fn[0] == '\0') {
        fprintf (stderr, "\n** ERROR: Output dose not specified in configuration file!\n");
        exit (0);
    }
}
