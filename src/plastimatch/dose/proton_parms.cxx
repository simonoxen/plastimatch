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
#include "plmsys.h"
#include "plm_math.h"
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
            if (sscanf (val, "%lf %lf %lf", &(scene->beam->src[0]), &(scene->beam->src[1]), &(scene->beam->src[2])) != 3) {
                goto error_exit;
            }
        }
        else if (!strcmp (key, "isocenter")) {
            if (sscanf (val, "%lf %lf %lf", &(scene->beam->isocenter[0]), &(scene->beam->isocenter[1]), &(scene->beam->isocenter[2])) != 3) {
                goto error_exit;
            }
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
            printf (">> %s\n", buf.c_str());
            handle_end_of_section (section);
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
            else if (buf.find ("[PEAK]") != std::string::npos
                || buf.find ("[peak]") != std::string::npos)
            {
                printf ("Found peak...\n");
                section = 3;
                continue;
            }
            else {
                printf ("Parse error: %s\n", buf_ori.c_str());
            }
        }

        printf ("buf = %s\n", buf.c_str());
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

    std::string foo = string_format ("Hello %s", "world");
    printf ("%s = %s\n", "Foobar", foo.c_str());
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
