/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */

/* -----------------------------------------------------------------------
   ion_sobp (Ion Spread Out Bragg Peak) is a class that creates a sobp from minimal and
   maximal depth to be targeted, or minimal and maximal energies of the
   pristine Bragg Peaks used to create the sobp. This class return the 
   weights of each pristine peak or the created sobp. It contains also the
   optimization process to get the weight of the pristine peaks. Mother class from which
   will derive the class for proton and other ions.
   ----------------------------------------------------------------------- */

#ifndef _photon_sobp_h_
#define _photon_sobp_h_

#include <stdio.h>
#include <vector>
#include "plmdose_config.h"
#include "plm_config.h"
#include "smart_pointer.h"

class Photon_depth_dose;
class Photon_sobp_private;

class PLMDOSE_API Photon_sobp {
public:
    SMART_POINTER_SUPPORT (Photon_sobp);
    Photon_sobp_private *d_ptr;
public:
    Photon_sobp ();
    ~Photon_sobp ();

    void set_resolution (double dres, int num_samples);

    /* Add a pristine peak to a sobp */
    void add (Photon_depth_dose* pristine_peak);
    void add (double E0, double spread, double dres, double dmax, 
        double weight);

    /* Set the min & max depth for automatic sobp optimization */
    void set_prescription_min_max (float d_min, float d_max);

    /* Save the depth dose to a file */
    void dump (const char* dir);

    /* Optimize, then generate sobp depth curve from prescription 
       range and modulation */
    void optimize ();

    /* Compute the sobp depth dose curve from weighted pristine peaks */
    bool generate ();

    /* Return simple depth dose result at depth */
    float lookup_energy (float depth);

    /* Return zmax */
    float get_maximum_depth();

    /* Print the parameters of the sobp */
    void printparameters();

    /* print sobp curve */
    void print_sobp_curve();

    /* set the minimal and maximal energy to buld the sobp peak */
    void SetMinMaxEnergies(int new_E_min, int new_E_max);
    /* set the minimal and maximal energy to buld the sobp peak and energy step */
    void SetMinMaxEnergies(int new_E_min, int new_E_max, int new_step); 
    /* set the minimal and maximal depth covered by the sobp */
    void SetMinMaxDepths(float new_z_min, float new_z_max);
    /* set the minimal and maximal depth covered by the sobp */
    void SetMinMaxDepths(float new_z_min, float new_z_max, float new_step);
    /* set the energy step only */
    void SetEnergyStep(int new_step);
    /* set energy step */
    void SetDepthStep(float new_step);	
	/* get peaks - not a pointer */
	std::vector<const Photon_depth_dose*> getPeaks();
    /* Weight optimizer */
    void Optimizer();

};

/* cost function used to optimize the sobp shape */
double cost_function_calculation_photon(std::vector<std::vector<double> > depth_dose, std::vector<double> weights, int num_peaks, int num_samples, std::vector<int> depth_in, std::vector<int> depth_out);

/* declaration of a matrix that contains the alpha and p parameters of the particles (Range = f(E, alpha, p) */
extern const double particle_parameters[][2];

#endif
