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
#include "rt_sobp_p.h"
#include "string_util.h"

Rt_sobp::Rt_sobp ()
{
    d_ptr = new Rt_sobp_private;
}

Rt_sobp::Rt_sobp (Particle_type part)
{
    d_ptr = new Rt_sobp_private(part);
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
Rt_sobp::add_peak ()
{
    /* To be implemented */
}

void
Rt_sobp::add_peak (Rt_depth_dose* depth_dose)
{
    d_ptr->depth_dose.push_back (depth_dose);

    /* GCS FIX: This should probably update the max depth too */
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
Rt_sobp::set_energyResolution(double eres)
{
    d_ptr->eres = eres;
    d_ptr->num_peaks = (int) ((d_ptr->E_max - d_ptr->E_min)/ d_ptr->eres + 1);
}

double 
Rt_sobp::get_energyResolution()
{
    return d_ptr->eres;
}

void
Rt_sobp::optimize ()
{	
    this->SetMinMaxDepths(
        d_ptr->prescription_dmin,
        d_ptr->prescription_dmax,
        d_ptr->dres);
    this->Optimizer2();
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
    printf("E_resolution : %d MeV \n",d_ptr->eres);
    printf("E_min : %d MeV \n",d_ptr->E_min);
    printf("E_max : %d MeV \n\n",d_ptr->E_max);
	
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
Rt_sobp::set_dres(double dres)
{
    d_ptr->dres = dres;
}

double
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
Rt_sobp::set_eres(int eres)
{
    d_ptr->eres = eres;
}

int 
Rt_sobp::get_eres()
{
    return d_ptr->eres;
}

void 
Rt_sobp::set_num_peaks(int num_peaks)
{
    d_ptr->num_peaks = num_peaks;
}

int 
Rt_sobp::get_num_peaks()
{
    return d_ptr->num_peaks;
}

void 
Rt_sobp::set_E_min(int E_min)
{
    d_ptr->E_min = E_min;
}

int 
Rt_sobp::get_E_min()
{
    return d_ptr->E_min;
}

void 
Rt_sobp::set_E_max(int E_max)
{
    d_ptr->E_max = E_max;
}

int 
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

void Rt_sobp::SetMinMaxEnergies(int new_E_min, int new_E_max) // set the sobp parameters by introducing the min and max energies
{
    if (new_E_max <= 0 || new_E_min <= 0)
    {
        printf("The energies min and max of the Sobp must be positive!\n");
        printf("Emin = %d, Emax = %d \n", new_E_min, new_E_max);
    }
    else
    {	
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

        d_ptr->dmin = ((10*d_ptr->alpha)*pow(d_ptr->E_min, d_ptr->p));
        d_ptr->dmax = ((10*d_ptr->alpha)*pow(d_ptr->E_max, d_ptr->p))+1;
        d_ptr->dend = d_ptr->dmax + 20;
        d_ptr->num_peaks = (int)(((d_ptr->E_max-d_ptr->E_min-1)/d_ptr->eres)+2);
        d_ptr->num_samples = (int)((d_ptr->dend/d_ptr->dres)+1);
        if ((d_ptr->num_samples-1)*d_ptr->dres < d_ptr->dend)
        {
            d_ptr->num_samples++;
        }

        if (d_ptr->d_lut) delete[] d_ptr->d_lut;
        d_ptr->d_lut = new float[d_ptr->num_samples];
        if (d_ptr->e_lut) delete[] d_ptr->e_lut;
        d_ptr->e_lut = new float[d_ptr->num_samples];


        for (int i = 0; i < d_ptr->num_samples-1; i++)
        {
            d_ptr->d_lut[i] = i*d_ptr->dres;
            d_ptr->e_lut[i] = 0;
        }

        d_ptr->d_lut[d_ptr->num_samples-1] = d_ptr->dend;
        d_ptr->e_lut[d_ptr->num_samples-1] = 0;
    }
}

void Rt_sobp::SetMinMaxEnergies(int new_E_min, int new_E_max, int new_step) // set the sobp parameters by introducing the min and max energies
{
    if (new_E_max <= 0 || new_E_min <= 0 || new_step < 0)
    {
        printf("The energies min and max of the Sobp must be positive and the step must be positive!\n");
        printf("Emin = %d, Emax = %d, step = %d \n", new_E_min, new_E_max, new_step);
    }
    else
    {	
        if (new_E_max >= new_E_min)
        {
            d_ptr->E_min = new_E_min;
            d_ptr->E_max = new_E_max;
            d_ptr->eres = new_step;
        }
        else
        {
            d_ptr->E_min = new_E_max;
            d_ptr->E_max = new_E_min;
            d_ptr->eres = new_step;
        }

        d_ptr->dmin = ((10*d_ptr->alpha)*pow(d_ptr->E_min, d_ptr->p));
        d_ptr->dmax = ((10*d_ptr->alpha)*pow(d_ptr->E_max, d_ptr->p))+1;
        d_ptr->dend = d_ptr->dmax + 20;
        d_ptr->num_peaks = (int)(((d_ptr->E_max-d_ptr->E_min-1)/d_ptr->eres)+2);
        d_ptr->num_samples = (int)((d_ptr->dend/d_ptr->dres)+1);
		
        if ((d_ptr->num_samples-1)*d_ptr->dres < d_ptr->dend)
        {
            d_ptr->num_samples++;
        }

        if (d_ptr->d_lut) delete[] d_ptr->d_lut;
        d_ptr->d_lut = new float[d_ptr->num_samples];
        if (d_ptr->e_lut) delete[] d_ptr->e_lut;
        d_ptr->e_lut = new float[d_ptr->num_samples];


        for (int i = 0; i < d_ptr->num_samples-1; i++)
        {
            d_ptr->d_lut[i] = i*d_ptr->dres;
            d_ptr->e_lut[i] = 0;
        }

        d_ptr->d_lut[d_ptr->num_samples-1] = d_ptr->dend;
        d_ptr->e_lut[d_ptr->num_samples-1] = 0;
    }
}

void Rt_sobp::SetMinMaxDepths(float new_z_min, float new_z_max) // set the sobp parameters by introducing the proximal and distal distances
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
        d_ptr->num_peaks = (int)(((d_ptr->E_max-d_ptr->E_min-1)/d_ptr->eres)+2);

        d_ptr->num_samples = (int)((d_ptr->dend/d_ptr->dres)+1);

        if ((d_ptr->num_samples-1)*d_ptr->dres < d_ptr->dend)
        {
            d_ptr->num_samples++;
        }

        if (d_ptr->d_lut) delete[] d_ptr->d_lut;
        d_ptr->d_lut = new float[d_ptr->num_samples];
        if (d_ptr->e_lut) delete[] d_ptr->e_lut;
        d_ptr->e_lut = new float[d_ptr->num_samples];

        for (int i = 0; i < d_ptr->num_samples-1; i++)
        {
            d_ptr->d_lut[i] = i*d_ptr->dres;
            d_ptr->e_lut[i] = 0;
        }

        d_ptr->d_lut[d_ptr->num_samples-1] = d_ptr->dend;
        d_ptr->e_lut[d_ptr->num_samples-1] = 0;

    }
}

void Rt_sobp::SetMinMaxDepths(float new_z_min, float new_z_max, float new_step) // set the sobp parameters by introducing the proximal and distal distances
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

        d_ptr->E_min = int(pow((d_ptr->dmin/(10*d_ptr->alpha)),(1/d_ptr->p)));
        d_ptr->E_max = int(pow((d_ptr->dmax/(10*d_ptr->alpha)),(1/d_ptr->p)))+d_ptr->eres;
        d_ptr->dend = d_ptr->dmax + 20;
        d_ptr->num_peaks = (int)(((d_ptr->E_max-d_ptr->E_min-1)/d_ptr->eres)+2);

        d_ptr->num_samples = (int)((d_ptr->dend/d_ptr->dres)+1);

        if ((d_ptr->num_samples-1)*d_ptr->dres < d_ptr->dend)
        {
            d_ptr->num_samples++;
        }

        if (d_ptr->d_lut) delete[] d_ptr->d_lut;
        d_ptr->d_lut = new float[d_ptr->num_samples];
        if (d_ptr->e_lut) delete[] d_ptr->e_lut;
        d_ptr->e_lut = new float[d_ptr->num_samples];

        for (int i = 0; i < d_ptr->num_samples-1; i++)
        {
            d_ptr->d_lut[i] = i*d_ptr->dres;
            d_ptr->e_lut[i] = 0;
        }

        d_ptr->d_lut[d_ptr->num_samples-1] = d_ptr->dend;
        d_ptr->e_lut[d_ptr->num_samples-1] = 0;
		
    }
}

void Rt_sobp::SetEnergyStep(int new_step)
{
    SetMinMaxEnergies(d_ptr->E_min, d_ptr->E_max, new_step);
}

void Rt_sobp::SetDepthStep(float new_step)
{
    SetMinMaxDepths(d_ptr->dmin, d_ptr->dmax, new_step);
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
