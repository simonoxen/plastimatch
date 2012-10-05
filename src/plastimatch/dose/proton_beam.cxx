/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plmdose_config.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include "bragg_curve.h"
#include "proton_beam.h"
#include "proton_sobp.h"

class Proton_beam_private {
public:
    Proton_beam_private ()
    {
        this->source[0] = -1000.f;
        this->source[1] = 0.f;
        this->source[2] = 0.f;
        this->isocenter[0] = 0.f;
        this->isocenter[1] = 0.f;
        this->isocenter[2] = 0.f;
//    this->beam_res = 1.f;
    }
public:
    double source[3];
    double isocenter[3];
};


Proton_Beam::Proton_Beam ()
{
    this->d_ptr = new Proton_beam_private;

    this->sobp = new Proton_sobp;

    this->E0 = 0.0;
    this->spread = 0.0;
    this->dres = 1.0;
    this->dmax = 0.0;
    this->num_samples = 0;
    this->weight = 1.0;
}

Proton_Beam::~Proton_Beam ()
{
    delete this->d_ptr;
    delete this->sobp;
}

bool
Proton_Beam::load (const char* fn)
{
    FILE* fp = fopen (fn, "r");
    char linebuf[128];

    if (!fp) {
        return false;
    }

    fgets (linebuf, 128, fp);
    fclose (fp);

    if (!strncmp (linebuf, "00001037", strlen ("00001037"))) {
        return this->load_xio (fn);
    } else {
        return this->load_txt (fn);
    }
}

const double*
Proton_Beam::get_source_position ()
{
    return d_ptr->source;
}

double
Proton_Beam::get_source_position (int dim)
{
    return d_ptr->source[dim];
}

void
Proton_Beam::set_source_position (const float* position)
{
    for (int d = 0; d < 3; d++) {
        d_ptr->source[d] = position[3];
    }
}

void
Proton_Beam::set_source_position (const double* position)
{
    for (int d = 0; d < 3; d++) {
        d_ptr->source[d] = position[3];
    }
}

const double*
Proton_Beam::get_isocenter_position ()
{
    return d_ptr->isocenter;
}

double
Proton_Beam::get_isocenter_position (int dim)
{
    return d_ptr->isocenter[dim];
}

void
Proton_Beam::set_isocenter_position (const float* position)
{
    for (int d = 0; d < 3; d++) {
        d_ptr->isocenter[d] = position[3];
    }
}

void
Proton_Beam::set_isocenter_position (const double* position)
{
    for (int d = 0; d < 3; d++) {
        d_ptr->isocenter[d] = position[3];
    }
}

bool
Proton_Beam::load_xio (const char* fn)
{
#if defined (commentout)
    int i, j;
    char* ptoken;
    char linebuf[128];
    FILE* fp = fopen (fn, "r");

    // Need to check for a magic number (00001037) here?
    
    /* skip the first 4 lines */
    for (i=0; i<4; i++) {
        fgets (linebuf, 128, fp);
    }

    /* line 5 contains the # of samples */
    fgets (linebuf, 128, fp);
    sscanf (linebuf, "%i", &this->num_samples);

    this->d_lut = (float*)malloc (this->num_samples*sizeof(float));
    this->e_lut = (float*)malloc (this->num_samples*sizeof(float));
    
    memset (this->d_lut, 0, this->num_samples*sizeof(float));
    memset (this->e_lut, 0, this->num_samples*sizeof(float));

    /* load in the depths (10 samples per line) */
    for (i=0, j=0; i<(this->num_samples/10)+1; i++) {
        fgets (linebuf, 128, fp);
        ptoken = strtok (linebuf, ",\n\0");
        while (ptoken) {
            this->d_lut[j++] = (float) strtod (ptoken, NULL);
            ptoken = strtok (NULL, ",\n\0");
        }
    }
    this->dmax = this->d_lut[j-1];

    /* load in the energies (10 samples per line) */
    for (i=0, j=0; i<(this->num_samples/10)+1; i++) {
        fgets (linebuf, 128, fp);
        ptoken = strtok (linebuf, ",\n\0");
        while (ptoken) {
            this->e_lut[j] = (float) strtod (ptoken, NULL);
            ptoken = strtok (NULL, ",\n\0");
            j++;
        }
    }

    fclose (fp);
#endif
    return true;
}

bool
Proton_Beam::load_txt (const char* fn)
{
#if defined (commentout)
    char linebuf[128];
    FILE* fp = fopen (fn, "r");

    while (fgets (linebuf, 128, fp)) {
        float range, dose;

        if (2 != sscanf (linebuf, "%f %f", &range, &dose)) {
            break;
        }

        this->num_samples++;
        this->d_lut = (float*) realloc (
                        this->d_lut,
                        this->num_samples * sizeof(float));

        this->e_lut = (float*) realloc (
                        this->e_lut,
                        this->num_samples * sizeof(float));

        this->d_lut[this->num_samples-1] = range;
        this->e_lut[this->num_samples-1] = dose;
        this->dmax = range;         /* Assume entries are sorted */
    }

    fclose (fp);
#endif
    return true;
}

void
Proton_Beam::add_peak ()
{
    this->sobp->add (this->E0, this->spread, this->dres, this->dmax, 
        this->weight);
}

float
Proton_Beam::lookup_energy (
    float depth
)
{
    return this->sobp->lookup_energy(depth);
}

bool
Proton_Beam::generate ()
{
    return this->sobp->generate ();
}

void
Proton_Beam::dump (const char* fn)
{
    this->sobp->dump (fn);
}
