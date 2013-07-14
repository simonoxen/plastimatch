/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plmdose_config.h"
#include <string>
#include <utility>
#include <vector>
#include <stdio.h>

#include "file_util.h"
#include "path_util.h"
#include "print_and_exit.h"
#include "ion_pristine_peak.h"
#include "ion_sobp.h"
#include "ion_sobp_optimize.h"
#include "string_util.h"

class Ion_sobp_private {
public:
    Ion_sobp_private () {
        d_lut = 0;
        e_lut = 0;
        dres = 0.0;
        dmax = 0.0;
        num_samples = 0;
        prescription_dmin = 0.f;
        prescription_dmax = 0.f;
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
    double dmax;                    /* maximum depth in SOBP curve (mm) */
    int num_samples;

    float prescription_dmin;
    float prescription_dmax;
};

Ion_sobp::Ion_sobp ()
{
    d_ptr = new Ion_sobp_private;
}

Ion_sobp::~Ion_sobp ()
{
    delete d_ptr;
}

void
Ion_sobp::add (const Ion_pristine_peak* pristine_peak)
{
    d_ptr->peaks.push_back (pristine_peak);

    /* GCS FIX: This should probably update the max depth too */
}

void
Ion_sobp::add (double E0, double spread, double dres, double dmax, 
    double weight)
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
    Ion_sobp_optimize op;
    double energy_step = 1.0;
    double energy_spread = 1.0;
    double depth_res = 1.0;

    op.SetMinMaxDepths (
        d_ptr->prescription_dmin,
        d_ptr->prescription_dmax,
        energy_step);
    op.Optimizer();

    /* In the future, the optimizer should use Ion_pristine_peak, 
       which would allow one to optimize user-defined depth-dose curves.
       For example, user-selected "spread" is ignored. */
    double max_depth = (double) op.m_z_end;
    for (int i = 0; i < op.m_EnergyNumber; i++) {
        double energy = d_ptr->prescription_dmin + i * energy_step;
        double weight = op.m_weights[i];
        this->add (energy, energy_spread, depth_res, 
            max_depth, weight);
    }

    this->generate ();
}

double
Ion_sobp::get_maximum_depth ()
{
    return d_ptr->dmax;
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
    for (i = 0; i < d_ptr->num_samples; i++) {
        if (d_ptr->d_lut[i] > depth) {
            i--;
            break;
        } else if (d_ptr->d_lut[i] == depth) {
            return d_ptr->e_lut[i];
        }
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
        if (!d_ptr->d_lut) {
            d_ptr->num_samples = ppp->num_samples;
            d_ptr->dres = ppp->dres;
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
Ion_sobp::dump (const char* fn)
{
    /* Dump SOBP */
    FILE* fp = fopen (fn, "w");
    for (int i=0; i < d_ptr->num_samples; i++) {
       fprintf (fp, "%3.2f %3.2f\n", d_ptr->d_lut[i], d_ptr->e_lut[i]);
    }
    fclose (fp);

    /* Dump pristine peaks */
    std::string dirname = file_util_dirname_string (fn);
    std::vector<const Ion_pristine_peak*>::const_iterator it 
        = d_ptr->peaks.begin();
    while (it != d_ptr->peaks.end ()) {
        std::string fn = string_format ("%s/pristine_%4.2f.txt",
            dirname.c_str(), (float) (*it)->E0);
        (*it)->dump (fn.c_str());
        it++;
    }
}
