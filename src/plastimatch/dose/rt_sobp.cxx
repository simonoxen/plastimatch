/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plmdose_config.h"
#include <string>
#include <utility>
#include <vector>
#include <stdio.h>
#include <math.h>

#include "bragg_curve.h"
#include "file_util.h"
#include "path_util.h"
#include "print_and_exit.h"
#include "rt_depth_dose.h"
#include "rt_sobp.h"
#include "string_util.h"

class Rt_sobp_private {
public:
    std::vector<const Rt_depth_dose*> depth_dose;

    float* d_lut;               /* depth array (mm) */
    float* e_lut;               /* energy array (MeV) */
    float* f_lut;		/* integrated energy array (MeV) */
    float dose_max;
    int num_samples;	        /* number of depths */
    int num_peaks;		/* number of peaks */

    std::vector<double> sobp_weight;

    float E_min;		/* lower energy */
    float E_max;		/* higher energy */
    float eres;			/* energy resolution */

    float dmin;			/* lower depth */
    float dmax;			/* higher depth */
    double dres;                /* depth resolution */
    float dend;			/* end of the depth array */

    /* p & alpha are parameters that bind depth and energy according to ICRU */
    Particle_type particle_type;
    double p;			
    double alpha;

    float prescription_dmin;
    float prescription_dmax;

public:
    Rt_sobp_private ()
    {
        d_lut = new float[0];
        e_lut = new float[0];
        f_lut = new float[0];
        dose_max = 1.f;
        num_samples = 0;
        E_min = 0.f;
        E_max = 0.f;
        eres = 1.0;
        dmin = 0.0;
        dmax = 0.0;
        dres = .01;
        dend = 0.0;
        set_particle_type (PARTICLE_TYPE_P);
        prescription_dmin = 50.f;
        prescription_dmax = 100.f;
    }
    Rt_sobp_private (Particle_type)
    {
        d_lut = new float[0];
        e_lut = new float[0];
        f_lut = new float[0];
        dres = .01;
        dose_max = 1.f;
        num_samples = 0;
        eres = 1.0;
        E_min = 0;
        E_max = 0;
        dmin = 0.0;
        dmax = 0.0;
        dend = 0.0;
        prescription_dmin = 50.f;
        prescription_dmax = 100.f;
        set_particle_type (PARTICLE_TYPE_P);
    }
    Rt_sobp_private (const Rt_sobp_private* rsp)
    {
        d_lut = new float[0];
        e_lut = new float[0];
        f_lut = new float[0];
        dres = rsp->dres;
        dose_max = 1.f;
        num_samples = rsp->num_samples;
        eres = rsp->eres;
        E_min = rsp->E_min;
        E_max = rsp->E_max;
        dmin = rsp->dmin;
        dmax = rsp->dmax;
        dend = rsp->dend;
        prescription_dmin = rsp->prescription_dmin;
        prescription_dmax = rsp->prescription_dmax;
        set_particle_type (rsp->particle_type);
    }
    ~Rt_sobp_private ()
    {
        if (d_lut) delete[] d_lut;
        if (e_lut) delete[] e_lut;
        if (f_lut) delete[] f_lut;
        clear_peaks ();
    }
public:
    void set_particle_type (Particle_type)
    {
        this->particle_type = particle_type;
        switch (particle_type) {
        case PARTICLE_TYPE_P:
            alpha = particle_parameters[0][0];
            p = particle_parameters[0][1];
            break;
        case PARTICLE_TYPE_HE:
            alpha = particle_parameters[1][0];
            p = particle_parameters[1][1];
            lprintf ("data for helium particle are not available - based on proton beam data");
            break;
        case PARTICLE_TYPE_LI:
            alpha = particle_parameters[2][0];
            p = particle_parameters[2][1];
            lprintf ("data for lithium particle type are not available - based on proton beam data");
            break;
        case PARTICLE_TYPE_BE:
            alpha = particle_parameters[3][0];
            p = particle_parameters[3][1];
            lprintf ("data for berilium particle type are not available - based on proton beam data");
            break;
        case PARTICLE_TYPE_B:
            alpha = particle_parameters[4][0];
            p = particle_parameters[4][1];
            lprintf ("data for bore particle type are not available - based on proton beam data");
            break;
        case PARTICLE_TYPE_C:
            alpha = particle_parameters[5][0];
            p = particle_parameters[5][1];
            lprintf ("data for carbon particle type are not available - based on proton beam data");
            break;
        case PARTICLE_TYPE_O:
            alpha = particle_parameters[7][0];
            p = particle_parameters[7][1];
            lprintf ("data for oxygen particle type are not available - based on proton beam data");
            break;
        default:
            alpha = particle_parameters[0][0];
            p = particle_parameters[0][1];
            particle_type = PARTICLE_TYPE_P;
            lprintf ("particle not found - proton beam chosen");
            break;
        }
    }
    void clear_peaks ()
    {
        std::vector<const Rt_depth_dose*>::iterator it;
        for (it = depth_dose.begin(); it != depth_dose.end(); ++it) {
            delete *it;
        }
        depth_dose.clear();
        sobp_weight.clear();
    }
};

Rt_sobp::Rt_sobp ()
{
    d_ptr = new Rt_sobp_private;
}

Rt_sobp::Rt_sobp (Particle_type part)
{
    d_ptr = new Rt_sobp_private(part);
}

Rt_sobp::Rt_sobp (const Rt_sobp::Pointer& rt_sobp)
{
    d_ptr = new Rt_sobp_private (rt_sobp->d_ptr);
}

Rt_sobp::~Rt_sobp ()
{
    delete d_ptr;
}

void
Rt_sobp::clear_peaks ()
{
    d_ptr->clear_peaks ();
}

void
Rt_sobp::add_peak (Rt_depth_dose* depth_dose)
{
    d_ptr->depth_dose.push_back (depth_dose);
    /* GCS FIX: This should probably update the max depth too - (MD Fix)*/
}

void
Rt_sobp::add_peak (double E0, double spread, 
    double dres, double dmax, double weight)
{
    switch(d_ptr->particle_type)
    {
    case PARTICLE_TYPE_P:			// proton
    {
        printf ("Adding peak to sobp (%f, %f, %f) [%f, %f]\n", 
            (float) E0, (float) spread, (float) weight,
            (float) dres, (float) dmax);
        Rt_depth_dose *depth_dose = new Rt_depth_dose (
            E0, spread, dres, dmax, weight);
        d_ptr->depth_dose.push_back (depth_dose);

        /* Update maximum */
        if (dmax > d_ptr->dmax) {
            d_ptr->dmax = dmax;
            break;
        }
    }
    case PARTICLE_TYPE_HE:			// helium
    {
        //to be implemented
    }
    break;
    case PARTICLE_TYPE_LI:			// lithium
    {
        //to be implemented
    }
    break;
    case PARTICLE_TYPE_BE:			// berilium
    {
        //to be implemented
    }
    break;
    case PARTICLE_TYPE_B:			// bore
    {
        //to be implemented
    }
    break;
    case PARTICLE_TYPE_C:			// carbon
    {
        //to be implemented
    }
    break;
    case PARTICLE_TYPE_N:			// nitrogen
    {
        //to be implemented
    }
    break;
    case PARTICLE_TYPE_O:			// oxygen
    {
        //to be implemented
    }
    break;
    default:
    {
        //to be implemented
    }
    }
}

void
Rt_sobp::set_prescription_min_max (float d_min, float d_max)
{
    d_ptr->prescription_dmin = d_min;
    d_ptr->prescription_dmax = d_max;
}

void
Rt_sobp::optimize ()
{	
    this->SetMinMaxDepths(
        d_ptr->prescription_dmin,
        d_ptr->prescription_dmax,
        d_ptr->dres);
    this->Optimizer ();
}

float
Rt_sobp::lookup_energy (
    float depth
)
{	
    int i;
    float energy = 0.0f;

    /* Sanity check */
    if (depth < 0) {
        return 0.0f;
    }

    /* Find index into profile arrays */
    for (i = 0; i < d_ptr->num_samples-1; i++) {
        if (d_ptr->d_lut[i] > depth) {
            i--;
            break;
        }
    }

    /* Clip input depth to maximum in lookup table */
    if (i == d_ptr->num_samples-1) {
        depth = d_ptr->d_lut[i];
    }

    /* Use index to lookup and interpolate energy */
    if (i >= 0 || i < d_ptr->num_samples) {
        // linear interpolation
        energy = d_ptr->e_lut[i]
            + (depth - d_ptr->d_lut[i])
            * ((d_ptr->e_lut[i+1] - d_ptr->e_lut[i]) 
                / (d_ptr->d_lut[i+1] - d_ptr->d_lut[i]));
    } else {
        // we wen't past the end of the lookup table
        energy = 0.0f;
    }
    return energy;   
}

bool
Rt_sobp::generate ()
{
    printf ("samples: %d\n", (int) d_ptr->depth_dose.size());
    std::vector<const Rt_depth_dose*>::const_iterator it 
        = d_ptr->depth_dose.begin();
    while (it != d_ptr->depth_dose.end ()) {
        const Rt_depth_dose *ppp = *it;

        /* Construct the data structure first time through */
        if (!d_ptr->d_lut || d_ptr->num_samples != ppp->num_samples) {
            d_ptr->num_samples = ppp->num_samples;
            d_ptr->dres = ppp->dres;

	    if (d_ptr->d_lut) delete[] d_ptr->d_lut;
	    if (d_ptr->e_lut) delete[] d_ptr->e_lut;

            d_ptr->d_lut = new float [ppp->num_samples];
            d_ptr->e_lut = new float [ppp->num_samples];

            for (int i = 0; i < d_ptr->num_samples; i++) {
                d_ptr->d_lut[i] = ppp->d_lut[i];
                d_ptr->e_lut[i] = 0;
            }
        }

        /* Check that this peak has the same num_samples, dres */
        if (ppp->num_samples != d_ptr->num_samples) {
            print_and_exit ("Error, mismatch in num_samples of SOBP\n");
        }
        if (ppp->dres != d_ptr->dres) {
            print_and_exit ("Error, mismatch in dres of SOBP\n");
        }

        /* Add weighted pristine peak to sobp */
        for (int i = 0; i < d_ptr->num_samples; i++) {
            d_ptr->e_lut[i] += ppp->weight * ppp->e_lut[i];
        }

        /* Go on to next pristine peak */
        it++;
    }
    return true;
}

void
Rt_sobp::dump (const char* dir)
{
    std::string dirname = dir;

    /* Dump SOBP */
    std::string sobp_fn = string_format ("%s/bragg_curve.txt", dir);
    FILE* fp = fopen (sobp_fn.c_str(), "w");
    for (int i=0; i < d_ptr->num_samples; i++) {
        fprintf (fp, "%3.2f %3.2f\n", d_ptr->d_lut[i], d_ptr->e_lut[i]);
    }
    fclose (fp);

    /* Dump pristine peaks */
    std::vector<const Rt_depth_dose*>::const_iterator it 
        = d_ptr->depth_dose.begin();
    while (it != d_ptr->depth_dose.end ()) {
        std::string fn = string_format ("%s/pristine_%4.2f.txt", dir, 
            (float) (*it)->E0);
        (*it)->dump (fn.c_str());
        it++;
    }
}

void 
Rt_sobp::SetParticleType (Particle_type particle_type)
{
    d_ptr->set_particle_type (particle_type);
    if (d_ptr->dmin !=0 && d_ptr->dmax !=0) {
        // we redefined the energies used to create the sobp
        SetMinMaxDepths(d_ptr->dmin, d_ptr->dmax);
    }
}

// return on the command line the parameters of the sobp to be build
void Rt_sobp::printparameters()
{
    printf ("Particle type : %s\n", 
        particle_type_string (d_ptr->particle_type));
		
    printf("\nNumber of peaks : %d\n",d_ptr->num_peaks);
    printf("E_resolution : %g MeV \n",d_ptr->eres);
    printf("E_min : %g MeV \n",d_ptr->E_min);
    printf("E_max : %g MeV \n\n",d_ptr->E_max);
	
    printf("z_resolution : %3.2f mm \n",d_ptr->dres);
    printf("z_min : %3.2f mm\n",d_ptr->dmin);
    printf("z_max : %3.2f mm\n",d_ptr->dmax);
    printf("z_end : %3.2f mm\n\n",d_ptr->dend);
}

void 
Rt_sobp::set_dose_lut(float* d_lut, float* e_lut, int num_samples)
{
    for (int i = 0; i < num_samples-1; i++)
    {
        d_ptr->d_lut[i] = d_lut[i];
        d_ptr->e_lut[i] = e_lut[i];
        if (i == 0) 
        {
            d_ptr->f_lut[i] = e_lut[i];
        } 
        else 
        {
            d_ptr->f_lut[i] = d_ptr->f_lut[i-1] + e_lut[i];
        }
    }
    d_ptr->num_samples = num_samples;
}

float* 
Rt_sobp::get_d_lut()
{
    return d_ptr->d_lut;
}

float* 
Rt_sobp::get_e_lut()
{
    return d_ptr->e_lut;
}

void 
Rt_sobp::set_dres(int dres)
{
    if (dres > 0)
    {
        d_ptr->dres = dres;
    }
    else {
        printf("***WARNING*** Depth resolution must be positive. Depth resolution unchanged");
    }
}

int
Rt_sobp::get_dres()
{
    return d_ptr->dres;
}

void 
Rt_sobp::set_num_samples(int num_samples)
{
    d_ptr->num_samples = num_samples;
}

int 
Rt_sobp::get_num_samples()
{
    return d_ptr->num_samples;
}

void 
Rt_sobp::set_energy_resolution (float eres)
{
    if (eres > 0)
    {
        d_ptr->eres = eres;
    }
    else {printf("***WARNING*** Energy resolution must be positive. Energy resolution unchanged");}
}

float
Rt_sobp::get_energy_resolution ()
{
    return d_ptr->eres;
}

size_t
Rt_sobp::get_num_peaks()
{
    return d_ptr->depth_dose.size();
}

void 
Rt_sobp::set_E_min(float E_min)
{
    d_ptr->E_min = E_min;
}

float
Rt_sobp::get_E_min()
{
    return d_ptr->E_min;
}

void 
Rt_sobp::set_E_max(float E_max)
{
    d_ptr->E_max = E_max;
}

float 
Rt_sobp::get_E_max()
{
    return d_ptr->E_max;
}

void 
Rt_sobp::set_dmin(float dmin)
{
    d_ptr->dmin = dmin;
}

float 
Rt_sobp::get_dmin()
{
    return d_ptr->dmin;
}

void 
Rt_sobp::set_dmax(float dmax)
{
    d_ptr->dmax = dmax;
    float num_sample = dmax / (float) d_ptr->dres;
    if (num_sample - (float) ((int) num_sample) == 0)
    {
        d_ptr->num_samples = num_sample;
    }
    else
    {
        d_ptr->num_samples = num_sample + 1;
    }
}

float 
Rt_sobp::get_dmax()
{
    return d_ptr->dmax;
}

void 
Rt_sobp::set_dend(float dend)
{
    d_ptr->dend = dend;
}

float
Rt_sobp::get_dend()
{
    return d_ptr->dend;
}

void 
Rt_sobp::set_particle_type(Particle_type particle_type)
{
    d_ptr->particle_type = particle_type;
}

Particle_type
Rt_sobp::get_particle_type()
{
    return d_ptr->particle_type;
}

void 
Rt_sobp::set_p(double p)
{
    d_ptr->p = p;
}

double 
Rt_sobp::get_p()
{
    return d_ptr->p;
}

void 
Rt_sobp::set_alpha(double alpha)
{
    d_ptr->alpha = alpha;
}

double 
Rt_sobp::get_alpha()
{
    return d_ptr->alpha;
}

void 
Rt_sobp::set_prescription_min(float prescription_min)
{
    d_ptr->prescription_dmin = prescription_min;
}

float 
Rt_sobp::get_prescription_min()
{
    return d_ptr->prescription_dmin;
}

void 
Rt_sobp::set_prescription_max(float prescription_max)
{
    d_ptr->prescription_dmax = prescription_max;
}

float
Rt_sobp::get_prescription_max()
{
    return d_ptr->prescription_dmax;
}

void 
Rt_sobp::add_weight(double weight)
{
    d_ptr->sobp_weight.push_back(weight);
}

std::vector<double> 
Rt_sobp::get_weight()
{
    return d_ptr->sobp_weight;
}

std::vector<const Rt_depth_dose*> 
Rt_sobp::get_depth_dose()
{
    return d_ptr->depth_dose;
}

void 
Rt_sobp::add_depth_dose(const Rt_depth_dose* depth_dose)
{
    Rt_depth_dose* dose = new Rt_depth_dose;	
    for (int i = 0; i < depth_dose->num_samples; i++)
    {
        dose->e_lut[i] = depth_dose->e_lut[i];
        dose->f_lut[i] = depth_dose->f_lut[i];
        dose->d_lut[i] = depth_dose->d_lut[i];
        dose->dmax = depth_dose->dmax;
        dose->dres = depth_dose->dres;
        dose->E0 = depth_dose->E0;
        dose->num_samples = depth_dose->num_samples;
        dose->spread = depth_dose->spread;
    }
    d_ptr->depth_dose.push_back(dose);
}

void Rt_sobp::print_sobp_curve()
{
    printf("\n print sobp curve : \n");
    if (d_ptr->num_samples != 0)
    {
        for ( int i = 0; i < d_ptr->num_samples ; i++)
        {
            printf("\n %f : %f", d_ptr->d_lut[i], d_ptr->e_lut[i]);
        }
    }
    else
    {
        printf(" void sobp curve");
    }
    printf("\n");
}

// set the sobp parameters by introducing the min and max energies
void Rt_sobp::SetMinMaxEnergies (
    float new_E_min, float new_E_max)
{
    if (new_E_max <= 0 || new_E_min <= 0)
    {
        printf("The energies min and max of the Sobp must be positive!\n");
        printf("Emin = %g, Emax = %g \n", new_E_min, new_E_max);
        return;
    }

    if (new_E_max >= new_E_min)
    {
        d_ptr->E_min = new_E_min;
        d_ptr->E_max = new_E_max;
    }
    else
    {
        d_ptr->E_min = new_E_max;
        d_ptr->E_max = new_E_min;
    }

    d_ptr->dmin = (10*d_ptr->alpha)*pow((double)d_ptr->E_min, d_ptr->p);
    d_ptr->dmax = (10*d_ptr->alpha)*pow((double)d_ptr->E_max, d_ptr->p)+1;
    d_ptr->dend = d_ptr->dmax + 20;
    d_ptr->num_samples = (int)((d_ptr->dend/d_ptr->dres)+1);
    if ((d_ptr->num_samples-1)*d_ptr->dres < d_ptr->dend)
    {
        d_ptr->num_samples++;
    }

    if (d_ptr->d_lut) delete[] d_ptr->d_lut;
    d_ptr->d_lut = new float[d_ptr->num_samples];
    if (d_ptr->e_lut) delete[] d_ptr->e_lut;
    d_ptr->e_lut = new float[d_ptr->num_samples];
    if (d_ptr->f_lut) delete[] d_ptr->f_lut;
    d_ptr->f_lut = new float[d_ptr->num_samples];

    for (int i = 0; i < d_ptr->num_samples-1; i++)
    {
        d_ptr->d_lut[i] = i*d_ptr->dres;
        d_ptr->e_lut[i] = 0;
        d_ptr->f_lut[i] = 0;
    }

    d_ptr->d_lut[d_ptr->num_samples-1] = d_ptr->dend;
    d_ptr->e_lut[d_ptr->num_samples-1] = 0;
    d_ptr->f_lut[d_ptr->num_samples-1] = 0;
    d_ptr->num_peaks = floor((d_ptr->E_max - d_ptr->E_min) / d_ptr->eres) + 1;
}

// set the sobp parameters by introducing the min and max energies
void Rt_sobp::SetMinMaxEnergies (
    float new_E_min, float new_E_max, float new_step)
{
    if (new_E_max <= 0 || new_E_min <= 0 || new_step < 0)
    {
        printf("The energies min and max of the Sobp must be positive and the step must be positive!\n");
        printf("Emin = %g, Emax = %g, step = %g \n", new_E_min, new_E_max, new_step);
        return;
    }

    this->SetMinMaxEnergies (new_E_min, new_E_max);
}

// set the sobp parameters by introducing the proximal and distal distances
void Rt_sobp::SetMinMaxDepths (
    float new_z_min, float new_z_max) 
{

    if (new_z_max <= 0 || new_z_min <= 0)
    {
        printf("Error: The depth min and max of the Sobp must be positive!\n");
        printf("zmin = %f, zmax = %f\n", new_z_min, new_z_max);
    }
    else
    {	
        if (new_z_max >= new_z_min)
        {
            d_ptr->dmin = new_z_min;
            d_ptr->dmax = new_z_max;
        }
        else
        {
            d_ptr->dmin = new_z_max;
            d_ptr->dmax = new_z_min;
        }

        d_ptr->E_min = int(pow((d_ptr->dmin/(10*d_ptr->alpha)),(1/d_ptr->p)));
        d_ptr->E_max = int(pow((d_ptr->dmax/(10*d_ptr->alpha)),(1/d_ptr->p)))+1;
        d_ptr->dend = d_ptr->dmax + 20;
        d_ptr->num_samples = (int)((d_ptr->dend/d_ptr->dres)+1);

        if ((d_ptr->num_samples-1)*d_ptr->dres < d_ptr->dend)
        {
            d_ptr->num_samples++;
        }
        if (d_ptr->d_lut) delete[] d_ptr->d_lut;
        d_ptr->d_lut = new float[d_ptr->num_samples];
        if (d_ptr->e_lut) delete[] d_ptr->e_lut;
        d_ptr->e_lut = new float[d_ptr->num_samples];
		if (d_ptr->f_lut) delete[] d_ptr->f_lut;
        d_ptr->f_lut = new float[d_ptr->num_samples];

        for (int i = 0; i < d_ptr->num_samples-1; i++)
        {
            d_ptr->d_lut[i] = i*d_ptr->dres;
            d_ptr->e_lut[i] = 0;
            d_ptr->f_lut[i] = 0;
        }
        d_ptr->d_lut[d_ptr->num_samples-1] = d_ptr->dend;
        d_ptr->e_lut[d_ptr->num_samples-1] = 0;
        d_ptr->f_lut[d_ptr->num_samples-1] = 0;
    }
	d_ptr->num_peaks = (d_ptr->E_max - d_ptr->E_min) / d_ptr->eres + 1;
}

// set the sobp parameters by introducing the proximal and distal distances
void Rt_sobp::SetMinMaxDepths (
    float new_z_min, float new_z_max, float new_step)
{
    if (new_z_max <= 0 || new_z_min <= 0)
    {
        printf("Error: The depth min and max and the step of the Sobp must be positive!\n");
        printf("zmin = %f, zmax = %f and z_step = %f\n", new_z_min, new_z_max, new_step);
    }
    else
    {	
        if (new_z_max >= new_z_min)
        {
            d_ptr->dmin = new_z_min;
            d_ptr->dmax = new_z_max;
            d_ptr->dres = new_step;
        }
        else
        {
            d_ptr->dmin = new_z_max;
            d_ptr->dmax = new_z_min;
            d_ptr->dres = new_step;
        }

        d_ptr->E_min = pow((d_ptr->dmin/(10*d_ptr->alpha)),(1/d_ptr->p));
        d_ptr->E_max = pow((d_ptr->dmax/(10*d_ptr->alpha)),(1/d_ptr->p))+d_ptr->eres;
        d_ptr->dend = d_ptr->dmax + 20;
        d_ptr->num_samples = (int)((d_ptr->dend/d_ptr->dres)+1);

        if ((d_ptr->num_samples-1)*d_ptr->dres < d_ptr->dend)
        {
            d_ptr->num_samples++;
        }

        if (d_ptr->d_lut) delete[] d_ptr->d_lut;
        d_ptr->d_lut = new float[d_ptr->num_samples];
        if (d_ptr->e_lut) delete[] d_ptr->e_lut;
        d_ptr->e_lut = new float[d_ptr->num_samples];
        if (d_ptr->f_lut) delete[] d_ptr->f_lut;
        d_ptr->f_lut = new float[d_ptr->num_samples];

        for (int i = 0; i < d_ptr->num_samples-1; i++)
        {
            d_ptr->d_lut[i] = i*d_ptr->dres;
            d_ptr->e_lut[i] = 0;
            d_ptr->f_lut[i] = 0;
        }

        d_ptr->d_lut[d_ptr->num_samples-1] = d_ptr->dend;
        d_ptr->e_lut[d_ptr->num_samples-1] = 0;
        d_ptr->f_lut[d_ptr->num_samples-1] = 0;		
    }
}

void Rt_sobp::SetEnergyStep(float new_step)
{
    SetMinMaxEnergies(d_ptr->E_min, d_ptr->E_max, new_step);
}

void Rt_sobp::SetDepthStep(float new_step)
{
    SetMinMaxDepths(d_ptr->dmin, d_ptr->dmax, new_step);
}

void Rt_sobp::SetDoseMax(float dose_max)
{
    d_ptr->dose_max = dose_max;
}

float Rt_sobp::GetDoseMax()
{
    return d_ptr->dose_max;
}

float Rt_sobp::get_maximum_depth()
{
    return d_ptr->dmax;
}

std::vector<const Rt_depth_dose*>
Rt_sobp::getPeaks()
{
    return d_ptr->depth_dose;
}

void Rt_sobp::Optimizer ()
{
    d_ptr->num_peaks = (int)(((d_ptr->E_max - d_ptr->E_min) / d_ptr->eres) + 1);

    double depth_maximum = 0;
    int idx_maximum = 0;

    std::vector<int> energies (d_ptr->num_peaks,0);
    std::vector<double> weight (d_ptr->num_peaks, 0);
  
    std::vector<double> init_vector (d_ptr->num_samples, 0);
    std::vector< std::vector<double> > depth_dose (d_ptr->num_peaks, init_vector);

    printf("\n %d Mono-energetic BP used:\n", d_ptr->num_peaks);

    /* updating the energies in the table) */
    for (int i = 0; i < d_ptr->num_peaks; i++)
    {
        energies[i]= d_ptr->E_min + i * d_ptr->eres;
        printf("%d ", energies[i]);
    }
    printf("\n");

    for (int i = 0; i < d_ptr->num_peaks; i++)
    {
        for (int j = 0; j < d_ptr->num_samples; j++)
        {
            depth_dose[i][j] = bragg_curve_norm((double)energies[i],1,(double)d_ptr->d_lut[j]);
        }
    }
	
    for (int i = d_ptr->num_peaks -1 ; i >= 0; i--)
    {
        if (i == d_ptr->num_peaks - 1)
        {
            weight[i] = 1.0;
        }
        else
        {
            /* Find depth max in mm*/
            depth_maximum = (double) get_proton_depth_max(energies[i]) / (100 * d_ptr->dres);
            idx_maximum = (int) depth_maximum;

            if (depth_maximum - (double) ((int) depth_maximum) > 0.5 &&  idx_maximum < d_ptr->num_samples)
            {
                idx_maximum++;
            }
            weight[i] = 1.0 - d_ptr->e_lut[idx_maximum];
            if (weight[i] < 0)
            {
                weight[i] = 0;
            }
        }
        for (int j = 0; j < d_ptr->num_samples; j++)
        {
            d_ptr->e_lut[j] += weight[i] * depth_dose[i][j];
        }
    }

    double mean_sobp = 0;
    double mean_count = 0;
    for (int i = 0; i < d_ptr->num_samples; i++)
    {
        if (d_ptr->d_lut[i] >= d_ptr->dmin && d_ptr->d_lut[i] <= d_ptr->dmax)
        {
            mean_sobp += d_ptr->e_lut[i];
            mean_count++;
        }
    }
    if (mean_count == 0)
    {
        printf("***WARNING*** The dose is null in the target interval\n");
        return;
    }

    /* SOBP norm and reset the depth dose*/
    for (int j = 0; j< d_ptr->num_samples; j++)
    {
        d_ptr->e_lut[j] =0;
    }

    for(int i = 0; i < d_ptr->num_peaks; i++)
    {
        weight[i] = weight[i] / mean_sobp * mean_count;
        for (int j = 0; j < d_ptr->num_samples; j++)
        {
            d_ptr->e_lut[j] += weight[i] * depth_dose[i][j];
        }
    }

    while (!d_ptr->depth_dose.empty())
    {
        d_ptr->depth_dose.pop_back();
    }

    d_ptr->num_peaks = d_ptr->num_peaks;
    for(int i = 0; i < d_ptr->num_peaks; i++)
    {
        this->add_peak ((double)energies[i],1, d_ptr->dres, 
            (double)d_ptr->dend, weight[i]);
        d_ptr->sobp_weight.push_back(weight[i]);
    }

    /* look for the max */
    double dose_max = 0;
    for(int i = d_ptr->num_samples-1; i >=0; i--)
    {
        if (d_ptr->e_lut[i] > dose_max)
        {
            dose_max = d_ptr->e_lut[i];
        }
    }
    this->SetDoseMax(dose_max);
}
