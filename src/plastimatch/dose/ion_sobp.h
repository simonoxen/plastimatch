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

#ifndef _ion_sobp_h_
#define _ion_sobp_h_

#include <stdio.h>
#include <vector>
#include "plmdose_config.h"
#include "plm_config.h"
#include "smart_pointer.h"

enum Particle_type {PARTICLE_TYPE_P=1, PARTICLE_TYPE_HE=2, PARTICLE_TYPE_LI=3, PARTICLE_TYPE_BE=4, PARTICLE_TYPE_B=5, PARTICLE_TYPE_C=6, PARTICLE_TYPE_O=8};

class Ion_pristine_peak;
class Ion_sobp_private;

class PLMDOSE_API Ion_sobp {
public:
    SMART_POINTER_SUPPORT (Ion_sobp);
    Ion_sobp_private *d_ptr;
public:
    Ion_sobp ();
	Ion_sobp (Particle_type part);
    ~Ion_sobp ();

    void set_resolution (double dres, int num_samples);

	/* set the type of particle (proton, helium ions, carbon ions...)*/
    void SetParticleType(Particle_type particle_type);

    /* Add a pristine peak to a sobp */
    void add (Ion_pristine_peak* pristine_peak);
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
	std::vector<const Ion_pristine_peak*> getPeaks();
    /* Weight optimizer */
    void Optimizer();
	void Optimizer2();

};

/* cost function used to optimize the sobp shape */
double cost_function_calculation(std::vector<std::vector<double> > depth_dose, std::vector<double> weights, int num_peaks, int num_samples, std::vector<int> depth_in, std::vector<int> depth_out);

/* declaration of a matrix that contains the alpha and p parameters of the particles (Range = f(E, alpha, p) */
extern const double particle_parameters[][2];

/* declaration of a matrix that contains the depth of the max for each energy */
extern const int max_depth_proton[];

#endif
