/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plmdose_config.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include "bragg_curve.h"
#include "RTP_beam.h"

class RTP_beam_private {
public:
    double source[3];
    double isocenter[3];
    int detail;
    char flavor;
    char homo_approx;
    Particle_type part;

public:
    RTP_sobp::Pointer sobp;
    std::string debug_dir;

    float beamWeight;

    float prescription_d_min;
    float prescription_d_max;
    float proximal_margin;
    float distal_margin;

    float source_size;

#if defined (commentout)
    double E0;                      /* initial ion energy (MeV) */
    double spread;                  /* beam energy sigma (MeV) */
    double dres;                    /* spatial resolution of bragg curve (mm)*/
    double dmax;                    /* maximum w.e.d. (mm) */
    int num_samples;                /* # of discrete bragg curve samples */
    double weight;					/* weight of the beam */
#endif

public:
    RTP_beam_private (Particle_type part)
    {
        this->source[0] = -1000.f;
        this->source[1] = 0.f;
        this->source[2] = 0.f;
        this->isocenter[0] = 0.f;
        this->isocenter[1] = 0.f;
        this->isocenter[2] = 0.f;
        this->detail = 1;
        this->flavor = 'a';
        this->homo_approx = 'n';
        this->part = part;

        this->sobp = RTP_sobp::New(new RTP_sobp(part));

        this->debug_dir = "";

        this->beamWeight = 1.0f;

        this->prescription_d_min = 0.;
        this->prescription_d_max = 0.;
        this->proximal_margin = 0.;
        this->distal_margin = 0.;

        this->source_size = 0.0f;


#if defined (commentout)
        this->E0 = 0.0;
        this->spread = 0.0;
        this->dres = 1.0;
        this->dmax = 0.0;
        this->num_samples = 0;
        this->weight = 1.0;
#endif
    }
};

RTP_beam::RTP_beam ()
{
    this->d_ptr = new RTP_beam_private(PARTICLE_TYPE_P);
}

RTP_beam::RTP_beam (Particle_type part)
{
    this->d_ptr = new RTP_beam_private(part);
}

RTP_beam::~RTP_beam ()
{
    delete this->d_ptr;
}

bool
RTP_beam::load (const char* fn)
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
RTP_beam::get_source_position ()
{
    return d_ptr->source;
}

double
RTP_beam::get_source_position (int dim)
{
    return d_ptr->source[dim];
}

void
RTP_beam::set_source_position (const float* position)
{
    for (int d = 0; d < 3; d++) {
        d_ptr->source[d] = position[d];
    }
}

void
RTP_beam::set_source_position (const double* position)
{
    for (int d = 0; d < 3; d++) {
        d_ptr->source[d] = position[d];
    }
}

const double*
RTP_beam::get_isocenter_position ()
{
    return d_ptr->isocenter;
}

double
RTP_beam::get_isocenter_position (int dim)
{
    return d_ptr->isocenter[dim];
}

void
RTP_beam::set_isocenter_position (const float* position)
{
    for (int d = 0; d < 3; d++) {
        d_ptr->isocenter[d] = position[d];
    }
}

void
RTP_beam::set_isocenter_position (const double* position)
{
    for (int d = 0; d < 3; d++) {
        d_ptr->isocenter[d] = position[d];
    }
}

int
RTP_beam::get_detail (void) const
{
    return d_ptr->detail;
}

void
RTP_beam::set_detail (int detail)
{
    d_ptr->detail = detail;
}

char
RTP_beam::get_flavor (void) const
{
    return d_ptr->flavor;
}

void
RTP_beam::set_flavor (char flavor)
{
    d_ptr->flavor = flavor;
}

char 
RTP_beam::get_homo_approx () const
{
    return d_ptr->homo_approx;
}
    
void 
RTP_beam::set_homo_approx (char homo_approx)
{
  d_ptr->homo_approx = homo_approx;
}

float
RTP_beam::get_beamWeight (void) const
{
    return d_ptr->beamWeight;
}

void
RTP_beam::set_beamWeight (float beamWeight)
{
    d_ptr->beamWeight = beamWeight;
}

double 
RTP_beam::get_sobp_maximum_depth ()
{
    return d_ptr->sobp->get_maximum_depth ();
}

void
RTP_beam::set_proximal_margin (float proximal_margin)
{
    d_ptr->proximal_margin = proximal_margin;
    d_ptr->sobp->set_prescription_min_max (
        d_ptr->prescription_d_min - d_ptr->proximal_margin,
        d_ptr->prescription_d_max + d_ptr->distal_margin);
}

void
RTP_beam::set_distal_margin (float distal_margin)
{
    d_ptr->distal_margin = distal_margin;
    d_ptr->sobp->set_prescription_min_max (
        d_ptr->prescription_d_min - d_ptr->proximal_margin,
        d_ptr->prescription_d_max + d_ptr->distal_margin);
}

void 
RTP_beam::set_sobp_prescription_min_max (float d_min, float d_max)
{
    d_ptr->prescription_d_min = d_min;
    d_ptr->prescription_d_max = d_max;
    d_ptr->sobp->set_prescription_min_max (
        d_ptr->prescription_d_min - d_ptr->proximal_margin,
        d_ptr->prescription_d_max + d_ptr->distal_margin);
}

void
RTP_beam::set_source_size(float source_size)
{
	d_ptr->source_size = source_size;
}

float
RTP_beam::get_source_size()
{
	return d_ptr->source_size;
}

void
RTP_beam::set_debug (const std::string& dir)
{
    d_ptr->debug_dir = dir;
}

bool
RTP_beam::load_xio (const char* fn)
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
RTP_beam::load_txt (const char* fn)
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
RTP_beam::add_peak (
    double E0,                      /* initial ion energy (MeV) */
    double spread,                  /* beam energy sigma (MeV) */
    double dres,                    /* spatial resolution of bragg curve (mm)*/
    double dmax,                    /* maximum w.e.d. (mm) */
    double weight)
{
    d_ptr->sobp->add (E0, spread, dres, dmax, weight);
}

float
RTP_beam::lookup_sobp_dose (
    float depth
)
{
    return d_ptr->sobp->lookup_energy(depth);
}

void
RTP_beam::optimize_sobp ()
{
    d_ptr->sobp->optimize ();
}

bool
RTP_beam::generate ()
{
    return d_ptr->sobp->generate ();
}

void
RTP_beam::dump (const char* dir)
{
    d_ptr->sobp->dump (dir);
}

RTP_sobp::Pointer
RTP_beam::get_sobp()
{
	return d_ptr->sobp;
}
