/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plmdose_config.h"
#include <string>
#include <utility>
#include <vector>
#include <stdio.h>
#include <math.h>

#include "file_util.h"
#include "path_util.h"
#include "print_and_exit.h"
#include "ion_pristine_peak.h"
#include "ion_sobp.h"
#include "bragg_curve.h"
#include "string_util.h"

#include "vnl/algo/vnl_amoeba.h"
#include "vnl/vnl_cost_function.h"

/* vxl needs you to wrap the function within a class */
class cost_function : public vnl_cost_function
{
public:
	std::vector<std::vector<double> > depth_dose;
	std::vector<double> weights;
	std::vector<int> depth_in;
	int num_peaks;
	int num_samples;
	double z_end;
	std::vector<int> depth_out;

    virtual double f (vnl_vector<double> const& vnl_x) {
        /* vxl requires you using their own vnl_vector type, 
           therefore we copy into a standard C/C++ array. */
		for (int i=0; i < num_peaks; i++)
		{
			weights[i] =vnl_x[i];
		}
        return cost_function_calculation(depth_dose,weights, num_peaks, num_samples, depth_in, depth_out);
    }
};

class Ion_sobp_private {
public:
    Ion_sobp_private (Particle_type part) {
        d_lut = new float[0];
        e_lut = new float[0];
        dres = 1.0;
        num_samples = 0;
	eres = 1.0;
	num_peaks = 0;
	E_min = 0;
	E_max = 0;
	dmin = 0.0;
	dmax = 0.0;
	dend = 0.0;
	particle_type = part;
        prescription_dmin = 0.f;
        prescription_dmax = 0.f;

        switch(particle_type)
        {
            case 1:			// proton
	    {
	        alpha = particle_parameters[0][0];
	        p = particle_parameters[0][1];
	        break;
	    }
       	    case 2:			// helium
	    {
	        alpha = particle_parameters[1][0];
	        p = particle_parameters[1][1];
	        printf("data for helium particle are not available - based on proton beam data");
	        break;
	    }
	    case 3:			// lithium
	    {
	        alpha = particle_parameters[2][0];
	        p = particle_parameters[2][1];
	        printf("data for lithium particle type are not available - based on proton beam data");
	        break;
	    }
	    case 4:			// berilium
	    {
	        alpha = particle_parameters[3][0];
	        p = particle_parameters[3][1];
	        printf("data for berilium particle type are not available - based on proton beam data");
	        break;
	    }
	    case 5:			// bore
	    {
	        alpha = particle_parameters[4][0];
	        p = particle_parameters[4][1];
	        printf("data for bore particle type are not available - based on proton beam data");
	        break;
	    }
	    case 6:			// carbon
	    {
	        alpha = particle_parameters[5][0];
	        p = particle_parameters[5][1];
	        printf("data for carbon particle type are not available - based on proton beam data");
	        break;
	    }
	    case 8:			// oxygen
	    {
	        alpha = particle_parameters[7][0];
	        p = particle_parameters[7][1];
	        printf("data for oxygen particle type are not available - based on proton beam data");
	        break;
	    }
	    default:
	    {
	        alpha = particle_parameters[0][0];
	        p = particle_parameters[0][1];
	        particle_type = PARTICLE_TYPE_P;
	        printf("particle not found - proton beam chosen");
	        break;
	    }
        }
    }
    ~Ion_sobp_private () {
        if (d_lut) delete[] d_lut;
        if (e_lut) delete[] e_lut;
        /* GCS FIX: This leaks memory in "peaks" */
    }
public:
    std::vector<const Ion_pristine_peak*> peaks;

    float* d_lut;                   /* depth array (mm) */
    float* e_lut;                   /* energy array (MeV) */
    double dres;
    int num_samples;				/* number of depths */

    int eres;						/* energy resolution */
    int num_peaks;					/* number of peaks */

    std::vector<double> sobp_weight;

    int E_min;						/* lower energy */
    int E_max;						/* higher energy */

    float dmin;					/* lower depth */
    float dmax;					/* higher depth */
    float dend;					/* end of the depth array */

    Particle_type particle_type;
    double p;						/* p  & alpha are parameters that bind depth and energy according to ICRU */
    double alpha;

    float prescription_dmin;
    float prescription_dmax;
};

Ion_sobp::Ion_sobp ()
{
    d_ptr = new Ion_sobp_private(PARTICLE_TYPE_P);
}

Ion_sobp::Ion_sobp (Particle_type part)
{
    d_ptr = new Ion_sobp_private(part);
}


Ion_sobp::~Ion_sobp ()
{
    delete d_ptr;
}

void
Ion_sobp::add (Ion_pristine_peak* pristine_peak)
{
    d_ptr->peaks.push_back (pristine_peak);

    /* GCS FIX: This should probably update the max depth too */
}

void
Ion_sobp::add (double E0, double spread, double dres, double dmax, 
    double weight)
{
    switch(d_ptr->particle_type)
    {
        case PARTICLE_TYPE_P:			// proton
        {
            printf ("Adding peak to sobp (%f, %f, %f) [%f, %f]\n", 
		(float) E0, (float) spread, (float) weight,
		(float) dres, (float) dmax);
	    Ion_pristine_peak *peak = new Ion_pristine_peak (
		E0, spread, dres, dmax, weight);
		d_ptr->peaks.push_back (peak);

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
Ion_sobp::set_prescription_min_max (float d_min, float d_max)
{
    d_ptr->prescription_dmin = d_min;
    d_ptr->prescription_dmax = d_max;
}

void
Ion_sobp::optimize ()
{	
    this->SetMinMaxDepths(
        d_ptr->prescription_dmin,
        d_ptr->prescription_dmax,
        d_ptr->eres);
    this->Optimizer();
}

float
Ion_sobp::lookup_energy (
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
Ion_sobp::generate ()
{
    std::vector<const Ion_pristine_peak*>::const_iterator it 
        = d_ptr->peaks.begin();
    while (it != d_ptr->peaks.end ()) {
        const Ion_pristine_peak *ppp = *it;

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
Ion_sobp::dump (const char* dir)
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
    std::vector<const Ion_pristine_peak*>::const_iterator it 
        = d_ptr->peaks.begin();
    while (it != d_ptr->peaks.end ()) {
        std::string fn = string_format ("%s/pristine_%4.2f.txt", dir, 
            (float) (*it)->E0);
        (*it)->dump (fn.c_str());
        it++;
    }
}

void Ion_sobp::SetParticleType(Particle_type particle_type)
{
	switch(particle_type)
	{
		case PARTICLE_TYPE_P:			// proton
		{
		    d_ptr->particle_type = particle_type;
		    d_ptr->alpha = particle_parameters[0][0];
		    d_ptr->p = particle_parameters[0][1];
		    if (d_ptr->dmin !=0 && d_ptr->dmax !=0)
		    {
		    	SetMinMaxDepths(d_ptr->dmin, d_ptr->dmax); // we redefined the energies used to create the sobp
		    }
		}
		break;
		case PARTICLE_TYPE_HE:			// helium
		{
		    d_ptr->particle_type = particle_type;
		    d_ptr->alpha = particle_parameters[1][0]; //to be updated
		    d_ptr->p = particle_parameters[1][1]; //to be updated
		    if (d_ptr->dmin !=0 && d_ptr->dmax !=0)
		    {
			SetMinMaxDepths(d_ptr->dmin, d_ptr->dmax); // we redefined the energies used to create the sobp
		    }
		    printf("data for helium particle are not available - proton beam chosen\n");
		}
		break;
		case PARTICLE_TYPE_LI:			// lithium
		{
		    d_ptr->particle_type = particle_type;
		    d_ptr->alpha = particle_parameters[2][0]; //to be updated
		    d_ptr->p = particle_parameters[2][1]; //to be updated
		    if (d_ptr->dmin !=0 && d_ptr->dmax !=0)
		    {
		    	SetMinMaxDepths(d_ptr->dmin, d_ptr->dmax); // we redefined the energies used to create the sobp
		    }
		    printf("data for lithium particle type are not available - proton beam chosen\n");
		}
		break;
		case PARTICLE_TYPE_BE:			// berilium
		{
		    d_ptr->particle_type = particle_type;
		    d_ptr->alpha = particle_parameters[3][0]; //to be updated
		    d_ptr->p = particle_parameters[3][1]; //to be updated
		    if (d_ptr->dmin !=0 && d_ptr->dmax !=0)
		    {
		    	SetMinMaxDepths(d_ptr->dmin, d_ptr->dmax); // we redefined the energies used to create the sobp
		    }
		    printf("data for berilium particle type are not available - proton beam chosen\n");
		}
		break;
		case PARTICLE_TYPE_B:			// bore
		{
		    d_ptr->particle_type = particle_type;
		    d_ptr->alpha = particle_parameters[4][0]; //to be updated
		    d_ptr->p = particle_parameters[4][1]; //to be updated
		    if (d_ptr->dmin !=0 && d_ptr->dmax !=0)
		    {
		    	SetMinMaxDepths(d_ptr->dmin, d_ptr->dmax); // we redefined the energies used to create the sobp
		    }
		    printf("data for bore particle type are not available - proton beam chosen\n");
		}
		break;
		case PARTICLE_TYPE_C:			// carbon
		{
		    d_ptr->particle_type = particle_type;
		    d_ptr->alpha = particle_parameters[5][0]; //to be updated
		    d_ptr->p = particle_parameters[5][1]; //to be updated
		    if (d_ptr->dmin !=0 && d_ptr->dmax !=0)
		    {
		    	SetMinMaxDepths(d_ptr->dmin, d_ptr->dmax); // we redefined the energies used to create the sobp
		    }
		    printf("data for carbon particle type are not available - proton beam chosen\n");
		}
		break;
		case PARTICLE_TYPE_O:			// oxygen
		{
		    d_ptr->particle_type = particle_type;
		    d_ptr->alpha = particle_parameters[7][0]; //to be updated
		    d_ptr->p = particle_parameters[7][1]; //to be updated
		    if (d_ptr->dmin !=0 && d_ptr->dmax !=0)
		    {
		    	SetMinMaxDepths(d_ptr->dmin, d_ptr->dmax); // we redefined the energies used to create the sobp
		    }
		    printf("data for oxygen particle type are not available - proton beam chosen\n");
		}
		break;
		default:
		{
		    d_ptr->particle_type = particle_type;
		    d_ptr->alpha = particle_parameters[0][0]; //to be updated
		    d_ptr->p = particle_parameters[0][1]; //to be updated
		    if (d_ptr->dmin !=0 && d_ptr->dmax !=0)
		    {
			SetMinMaxDepths(d_ptr->dmin, d_ptr->dmax); // we redefined the energies used to create the sobp
		    }
		    printf("particle not found - proton beam chosen\n");
		}
	}
}

void Ion_sobp::printparameters()  // return on the command line the parameters of the sobp to be build
{
	printf("\nParticle type : ");
	switch(d_ptr->particle_type)
	{
		case PARTICLE_TYPE_P:			// proton
		{
			printf("Proton\n");
		}
		break;
		case PARTICLE_TYPE_HE:			// helium
		{
			printf("Helium\n");
		}
		break;
		case PARTICLE_TYPE_LI:			// lithium
		{
			printf("Lithium\n");
		}
		break;
		case PARTICLE_TYPE_BE:			// berilium
		{
			printf("Berillium\n");
		}
		break;
		case PARTICLE_TYPE_B:			// bore
		{
			printf("Bore\n");
		}
		break;
		case PARTICLE_TYPE_C:			// carbon
		{
			printf("Carbon\n");
		}
		break;
		case PARTICLE_TYPE_O:			// oxygen
		{
			printf("Oxygen\n");
		}
		break;
		default:
		{
			printf("particle_type not found - default : Proton \n");
		}
	}
	
		
	printf("\nNumber of peaks : %d\n",d_ptr->num_peaks);
	printf("E_resolution : %d MeV \n",d_ptr->eres);
	printf("E_min : %d MeV \n",d_ptr->E_min);
	printf("E_max : %d MeV \n\n",d_ptr->E_max);
	
	printf("z_resolution : %3.2f mm \n",d_ptr->dres);
	printf("z_min : %3.2f mm\n",d_ptr->dmin);
	printf("z_max : %3.2f mm\n",d_ptr->dmax);
	printf("z_end : %3.2f mm\n\n",d_ptr->dend);
}

void Ion_sobp::print_sobp_curve()
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

void Ion_sobp::SetMinMaxEnergies(int new_E_min, int new_E_max) // set the sobp parameters by introducing the min and max energies
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

void Ion_sobp::SetMinMaxEnergies(int new_E_min, int new_E_max, int new_step) // set the sobp parameters by introducing the min and max energies
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

void Ion_sobp::SetMinMaxDepths(float new_z_min, float new_z_max) // set the sobp parameters by introducing the proximal and distal distances
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

void Ion_sobp::SetMinMaxDepths(float new_z_min, float new_z_max, float new_step) // set the sobp parameters by introducing the proximal and distal distances
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

void Ion_sobp::SetEnergyStep(int new_step)
{
	SetMinMaxEnergies(d_ptr->E_min, d_ptr->E_max, new_step);
}

void Ion_sobp::SetDepthStep(float new_step)
{
	SetMinMaxDepths(d_ptr->dmin, d_ptr->dmax, new_step);
}

float Ion_sobp::get_maximum_depth()
{
	return d_ptr->dmax;
}

std::vector<const Ion_pristine_peak*>
Ion_sobp::getPeaks()
{
	return d_ptr->peaks;
}



void Ion_sobp::Optimizer() // the optimizer to get the optimized weights of the beams, optimized by a cost function (see below)
{
	/* Create function object (for function to be minimized) */
    cost_function cf;

	cf.num_samples = d_ptr->num_samples;
	cf.num_peaks = d_ptr->num_peaks;
	
	for (int i = 0; i < d_ptr->num_peaks; i++)
	{
		cf.weights.push_back(0);
	}
	
	std::vector<int> energies (d_ptr->num_peaks,0);
	std::vector<double> init_vector (d_ptr->num_samples,0);


	cf.depth_dose.push_back(init_vector);

	printf("\n %d Mono-energetic BP used: ", cf.num_peaks);

	energies[0]= d_ptr->E_min;
	printf("%d ", energies[0]);

	cf.depth_dose[0][0] = bragg_curve((double)energies[0],1,0);  // creation of the matrix gathering all the depth dose of the BP constituting the sobp

	for (int j = 0; j < d_ptr->num_samples; j++)
	{
		cf.depth_dose[0][j] = bragg_curve((double)energies[0],1,(double)d_ptr->d_lut[j]);
	}

	for (int i=1; i < cf.num_peaks-1; i++)
    {
		energies[i]=energies[i-1]+d_ptr->eres;
        printf("%d ",energies[i]);
		
		cf.depth_dose.push_back(init_vector);
		for (int j = 0; j < d_ptr->num_samples; j++)
		{
			cf.depth_dose[i][j] = bragg_curve(energies[i],1,d_ptr->d_lut[j]);
		}
    }

	energies[cf.num_peaks-1]= d_ptr->E_max;
	printf("%d \n", energies[cf.num_peaks-1]);

	cf.depth_dose.push_back(init_vector);
	for (int j = 0; j < d_ptr->num_samples; j++)
	{
		cf.depth_dose[cf.num_peaks-1][j] = bragg_curve(energies[cf.num_peaks-1],1,d_ptr->d_lut[j]);
	}


	for (int i = 0; i < d_ptr->num_samples ; i++) // creation of the two intervals that represents the inner part of the sobp and the outer part
    {
		cf.depth_in.push_back(0);
		cf.depth_out.push_back(0);

		if (d_ptr->d_lut[i]>=d_ptr->dmin && d_ptr->d_lut[i]<=d_ptr->dmax)
        {
                cf.depth_in[i] = 1;
                cf.depth_out[i] = 0;
        }
        else
        {
            cf.depth_in[i] = 0;
            cf.depth_out[i] = 1;
        }
    }	

	/* Create optimizer object */
    vnl_amoeba nm (cf);


    /* Set some optimizer parameters */
    nm.set_x_tolerance (0.0001);
    nm.set_f_tolerance (0.0000001);
    nm.set_max_iterations (1000000);

	/* Set the starting point */
	vnl_vector<double> x(cf.num_peaks, 0.05);
	const vnl_vector<double> y(cf.num_peaks, 0.5);

	/* Run the optimizer */
    nm.minimize (x,y);

	while (!d_ptr->peaks.empty())
	{
		d_ptr->peaks.pop_back();
	}

	for(int i = 0; i < d_ptr->num_peaks; i++)
	{
		this->add((double)energies[i],1, d_ptr->dres, (double)d_ptr->dend, cf.weights[i]);
		d_ptr->sobp_weight.push_back(cf.weights[i]);
	}

	d_ptr->num_samples = d_ptr->peaks[0]->num_samples;

	this->generate();
}

double cost_function_calculation(std::vector<std::vector<double> > depth_dose, std::vector<double> weights, int num_peaks, int num_samples, std::vector<int> depth_in, std::vector<int> depth_out) // cost function to be optimized in order to find the best weights and fit a perfect sobp
{
	std::vector<double> diff (num_samples, 0);
	std::vector<double> excess (num_samples, 0);
	std::vector<double> f (num_samples, 0);
	double f_tot = 0;
	double sobp_max = 0;
	double sum = 0;

	for (int j = 0; j < num_samples; j++) // we fit the curve on all the depth
    {
        sum = 0;
        for (int k = 0; k < num_peaks; k++)
        {
            sum = sum + weights[k]*depth_dose[k][j];
        }
        diff[j] = depth_in[j] * fabs(sum-30); // first parameters: the difference sqrt(standard deviation) between the curve and the perfect sobp, in the sobp area
        if (diff[j] > sobp_max)
        {
            sobp_max = diff[j];					// second parameters: the max difference between the curve and the perfect sobp, in the sobp area
        }

		excess[j] = depth_out[j] * (sum-30);// first parameters: the excess difference sqrt(standard deviation) between the curve and the perfect sobp, out of the sobp area (we want it far lower that the sobp flat region
        if (excess[j] < 0)
        {
             excess[j] = 0;
        }
        f[j]= 0.05 * diff[j]*diff[j] + 0.1 * excess[j] * excess[j]; // this 3 parameters are assessed, and weighted by 3 coefficient (to be optimized to get a beautiful sobp) and the value of the global function is returned
        f_tot = f_tot+f[j];
	}

	f_tot += 0.005 * sobp_max * num_samples;

	for(int i=0; i < num_peaks; i++)
	{
		if (weights[i] < 0)
		{
			f_tot = 2* f_tot;
		}
	}
	/*printf("\n f_tot = %lg", f_tot);
	for (int i = 0; i < num_peaks; i++)
	{
		printf (" %lg ", weights[i]);
	}*/

	return f_tot; //we return the fcost value
}

/* matrix that contains the alpha and p parameters Range = f(E, alpha, p)
    First line is proton, second line is He... */

extern const double particle_parameters[][2] = {
    0.0022, 1.77,   //P
    0.0022, 1.77,   //HE
    0.0022, 1.77,   //LI
    0.0022, 1.77,   //BE
    0.0022, 1.77,   //B
    0.0022, 1.77,   //C
    0.00,   0.00,   //N not used for ion therapy - set to 0
    0.0022, 1.77    //O
};