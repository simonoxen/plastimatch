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

Proton_Parms::Proton_Parms ()
{
    this->scene = new Proton_Scene;

    this->threading = THREADING_CPU_OPENMP;
    this->flavor = 'a';

    this->debug = 0;
    this->detail = 0;
    this->ray_step = 1.0f;
    this->scale = 1.0f;
    this->input_fn[0] = '\0';
    this->output_fn[0] = '\0';

    this->patient = NULL;
}

Proton_Parms::~Proton_Parms ()
{
    delete this->scene;
}

static void
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
Proton_Parms::set_key_val (
    const char* key, 
    const char* val, 
    int section
)
{
    Proton_Scene* scene = this->scene;
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
            scene->beam->load (val);
        }
        else if (!strcmp (key, "pos")) {
            float beam_source_position[3];
            int rc = sscanf (val, "%f %f %f", &beam_source_position[0],
                &beam_source_position[1], &beam_source_position[2]);
            if (rc != 3) {
                goto error_exit;
            }
            scene->beam->set_source_position (beam_source_position);
        }
        else if (!strcmp (key, "isocenter")) {
            float isocenter_position[3];
            int rc = sscanf (val, "%f %f %f", &isocenter_position[0],
                &isocenter_position[1], &isocenter_position[2]);
            if (rc != 3) {
                goto error_exit;
            }
            scene->beam->set_isocenter_position (isocenter_position);
        }
        else if (!strcmp (key, "energy")) {
            if (sscanf (val, "%lf", &(scene->beam->E0)) != 1) {
                goto error_exit;
            }
        }
        else if (!strcmp (key, "spread")) {
            if (sscanf (val, "%lf", &(scene->beam->spread)) != 1) {
                goto error_exit;
            }
        }
        else if (!strcmp (key, "depth")) {
            if (sscanf (val, "%lf", &(scene->beam->dmax)) != 1) {
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
            double offset;
            if (sscanf (val, "%lf", &offset) != 1) {
                goto error_exit;
            }
            scene->ap->set_offset (offset);
        }
        else if (!strcmp (key, "resolution")) {
            int ires[2];
            if (sscanf (val, "%i %i", &ires[0], &ires[1]) != 2) {
                goto error_exit;
            }
            scene->ap->set_dim (ires);
        }
        break;

        /* [PEAK] */
    case 3:
        if (!strcmp (key, "energy")) {
            if (sscanf (val, "%lf", &(scene->beam->E0)) != 1) {
                goto error_exit;
            }
        }
        else if (!strcmp (key, "spread")) {
            if (sscanf (val, "%lf", &(scene->beam->spread)) != 1) {
                goto error_exit;
            }
        }
        else if (!strcmp (key, "depth")) {
            if (sscanf (val, "%lf", &(scene->beam->dmax)) != 1) {
                goto error_exit;
            }
        }
        else if (!strcmp (key, "weight")) {
            if (sscanf (val, "%lf", &(scene->beam->weight)) != 1) {
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
Proton_Parms::handle_end_of_section (int section)
{
    switch (section) {
    case 0:
        /* Settings */
        break;
    case 1:
        /* Beam */
        break;
    case 2:
        /* Aperture */
        break;
    case 3:
        /* Peak */
        scene->beam->add_peak ();
        break;
    }
}

void
Proton_Parms::parse_config (
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
            handle_end_of_section (section);
            if (ci_find (buf, "[SETTINGS]") != std::string::npos)
            {
                section = 0;
                continue;
            }
            else if (ci_find (buf, "[BEAM]") != std::string::npos)
            {
                section = 1;
                continue;
            }
            else if (ci_find (buf, "[APERTURE]") != std::string::npos)
            {
                section = 2;
                continue;
            }
            else if (ci_find (buf, "[PEAK]") != std::string::npos) 
            {
                section = 3;
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

    handle_end_of_section (section);
}

bool
Proton_Parms::parse_args (int argc, char** argv)
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

    if (!scene->beam->generate ()) {
        return false;
    }

    if (this->output_fn[0] == '\0') {
        fprintf (stderr, "\n** ERROR: Output dose not specified in configuration file!\n");
        return false;
    }

    if (this->input_fn[0] == '\0') {
        fprintf (stderr, "\n** ERROR: Patient image not specified in configuration file!\n");
        return false;
    } else {
        /* load the patient and insert into the scene */
        this->patient = plm_image_load (this->input_fn, PLM_IMG_TYPE_ITK_FLOAT);
        if (!this->patient) {
            fprintf (stderr, "\n** ERROR: Unable to load patient volume.\n");
            return false;
        }
        this->scene->set_patient (this->patient);
    }

    /* try to setup the scene with the provided parameters */
    if (!this->scene->init (this->ray_step)) {
        fprintf (stderr, "ERROR: Unable to initilize scene.\n");
        return false;
    }

    return true;
}
