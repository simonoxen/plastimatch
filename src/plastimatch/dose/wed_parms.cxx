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

#include "plmbase.h"
#include "plmdose.h"

#include "plm_math.h"
#include "print_and_exit.h"
#include "string_util.h"

#ifndef NULL
#define NULL ((void*)0)
#endif

Wed_Parms::Wed_Parms ()
{
    this->scene = new Proton_Scene;

    this->debug = 0;
    this->ray_step = 1.0f;
    this->input_ct_fn[0] = '\0';
    this->input_dose_fn[0] = '\0';
    this->output_ct_fn[0] = '\0';
    this->output_dose_fn[0] = '\0';

    this->ct_vol = NULL;
    this->dose_vol = NULL;
}

Wed_Parms::~Wed_Parms ()
{
    delete this->scene;
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
    Proton_Scene* scene = this->scene;
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
            if (sscanf (val, "%lf %lf %lf", &(scene->beam->src[0]), &(scene->beam->src[1]), &(scene->beam->src[2])) != 3) {
                goto error_exit;
            }
        }
        else if (!strcmp (key, "isocenter")) {
            if (sscanf (val, "%lf %lf %lf", &(scene->beam->isocenter[0]), &(scene->beam->isocenter[1]), &(scene->beam->isocenter[2])) != 3) {
                goto error_exit;
            }
        }
        else if (!strcmp (key, "res")) {
            if (sscanf (val, "%lf", &(scene->beam->dres)) != 1) {
                goto error_exit;
            }
        }

        break;

    /* [APERTURE] */
    case 2:
        if (!strcmp (key, "up")) {
            if (sscanf (val, "%lf %lf %lf", &(scene->ap->vup[0]), &(scene->ap->vup[1]), &(scene->ap->vup[2])) != 3) {
                goto error_exit;
            }
        }
        else if (!strcmp (key, "center")) {
            if (sscanf (val, "%lf %lf", &(scene->ap->ic[0]), &(scene->ap->ic[1])) != 2) {
                goto error_exit;
            }
        }
        else if (!strcmp (key, "offset")) {
            if (sscanf (val, "%lf", &(scene->ap->ap_offset)) != 1) {
                goto error_exit;
            }
        }
        else if (!strcmp (key, "resolution")) {
            if (sscanf (val, "%i %i", &(scene->ap->ires[0]), &(scene->ap->ires[1])) != 2) {
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
    if (scene->beam->d_lut == NULL) {
        /* measured bragg curve not supplied, try to generate */
        if (!scene->beam->generate ()) {
            return false;
        }
    }
#endif
    // JAS 2012.08.10
    //   Hack so that I can reuse the proton code.  The values
    //   don't actually matter.
    scene->beam->E0 = 1.0;
    scene->beam->spread = 1.0;
    scene->beam->dmax = 1.0;

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
    } else {
        /* load the patient and insert into the scene */
        this->ct_vol = plm_image_load (this->input_ct_fn, PLM_IMG_TYPE_ITK_FLOAT);
        if (!this->ct_vol) {
            fprintf (stderr, "\n** ERROR: Unable to load patient volume.\n");
            return false;
        }
        this->scene->set_patient (this->ct_vol);
    }

    /* try to setup the scene with the provided parameters */
    if (!this->scene->init (this->ray_step)) {
        fprintf (stderr, "ERROR: Unable to initilize scene.\n");
        return false;
    }

    return true;
}
