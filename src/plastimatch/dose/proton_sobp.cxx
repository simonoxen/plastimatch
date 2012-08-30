/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plmdose_config.h"
#include <string>
#include <utility>
#include <vector>
#include <stdio.h>
#include "file_util.h"
#include "print_and_exit.h"
#include "proton_sobp.h"
#include "proton_pristine_peak.h"
#include "string_util.h"

class Proton_sobp_private {
public:
    Proton_sobp_private () {
        d_lut = 0;
        e_lut = 0;
        dres = 0.0;
        num_samples = 0;
    }
    ~Proton_sobp_private () {
        if (d_lut) delete[] d_lut;
        if (e_lut) delete[] e_lut;
        /* GCS FIX: This leaks memory in "peaks" */
    }
public:
    std::vector<const Proton_pristine_peak*> peaks;

    float* d_lut;                   /* depth array (mm) */
    float* e_lut;                   /* energy array (MeV) */
    double dres;
    int num_samples;
};

Proton_sobp::Proton_sobp ()
{
    d_ptr = new Proton_sobp_private;
}

Proton_sobp::~Proton_sobp ()
{
    delete d_ptr;
}

void
Proton_sobp::add (const Proton_pristine_peak* pristine_peak)
{
    d_ptr->peaks.push_back (pristine_peak);
}

void
Proton_sobp::add (double E0, double spread, double dres, double dmax, 
    double weight)
{
    printf ("Adding peak to sobp (%f, %f, %f) [%f, %f]\n", 
        (float) E0, (float) spread, (float) weight,
        (float) dres, (float) dmax);
    Proton_pristine_peak *peak = new Proton_pristine_peak (
        E0, spread, dres, dmax, weight);
    d_ptr->peaks.push_back (peak);
}

float
Proton_sobp::lookup_energy (
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
Proton_sobp::generate ()
{
    std::vector<const Proton_pristine_peak*>::const_iterator it 
        = d_ptr->peaks.begin();
    while (it != d_ptr->peaks.end ()) {
        const Proton_pristine_peak *ppp = *it;

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
Proton_sobp::dump (const char* fn)
{
    /* Dump SOBP */
    FILE* fp = fopen (fn, "w");
    for (int i=0; i < d_ptr->num_samples; i++) {
       fprintf (fp, "%3.2f %3.2f\n", d_ptr->d_lut[i], d_ptr->e_lut[i]);
    }
    fclose (fp);

    /* Dump pristine peaks */
    std::string dirname = file_util_dirname_string (fn);
    std::vector<const Proton_pristine_peak*>::const_iterator it 
        = d_ptr->peaks.begin();
    while (it != d_ptr->peaks.end ()) {
        std::string fn = string_format ("%s/pristine_%4.2f.txt",
            dirname.c_str(), (float) (*it)->E0);
        (*it)->dump (fn.c_str());
        it++;
    }
}
