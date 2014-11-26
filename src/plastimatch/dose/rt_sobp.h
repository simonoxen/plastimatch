/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
/* -----------------------------------------------------------------------
   rt_sobp (Ion Spread Out Bragg Peak) is a class that creates 
   a sobp from minimal and
   maximal depth to be targeted, or minimal and maximal energies of the
   pristine Bragg Peaks used to create the sobp. This class return the 
   weights of each pristine peak or the created sobp. It contains also the
   optimization process to get the weight of the pristine peaks. 
   Mother class from which
   will derive the class for proton and other ions.
   ----------------------------------------------------------------------- */
#ifndef _rt_sobp_h_
#define _rt_sobp_h_

#include "plmdose_config.h"
#include <stdio.h>
#include <vector>
#include "particle_type.h"
#include "plmdose_config.h"
#include "plm_config.h"
#include "rt_lut.h"
#include "smart_pointer.h"

class Rt_depth_dose;
class Rt_sobp_private;

class PLMDOSE_API Rt_sobp {
public:
    SMART_POINTER_SUPPORT (Rt_sobp);
    Rt_sobp_private *d_ptr;
public:
    Rt_sobp ();
    Rt_sobp (Particle_type part);
    Rt_sobp (Rt_sobp*& rt_sobp);
    ~Rt_sobp ();

    void set_resolution (double dres, int num_samples);
    void set_energyResolution(double eres);
    double get_energyResolution();

    /* set the type of particle (proton, helium ions, carbon ions...)*/
    void SetParticleType(Particle_type particle_type);

    /* Remove all peaks */
    void clear_peaks ();

    /* Add a pristine peak to a sobp */
    void add_peak ();
    void add_peak (Rt_depth_dose* depth_dose);
    void add_peak (double E0, double spread, 
        double dres, double dmax, double weight);

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

    /* Set private members */
    void set_dose_lut(float* d_lut, float* e_lut, int num_samples);
    float* get_d_lut();
    float* get_e_lut();
    void set_dres(double dres);
    double get_dres();
    void set_num_samples(int num_samples);
    int get_num_samples();
    void set_eres(int eres);
    int get_eres();
    void set_num_peaks(int num_peaks);
    int get_num_peaks();
    void set_E_min(int E_min);
    int get_E_min();
    void set_E_max(int E_max);
    int get_E_max();
    void set_dmin(float dmin);
    float get_dmin();
    void set_dmax(float dmax);
    float get_dmax();
    void set_dend(float dend);
    float get_dend();
    void set_particle_type(Particle_type particle_type);
    Particle_type get_particle_type();
    void set_p(double p);
    double get_p();
    void set_alpha(double alpha);
    double get_alpha();
    void set_prescription_min(float prescription_min);
    float get_prescription_min();
    void set_prescription_max(float prescription_max);
    float get_prescription_max();
    void add_weight(double sobp_weight);
    std::vector<double> get_weight();
    std::vector<const Rt_depth_dose*> get_depth_dose();
    void add_depth_dose(const Rt_depth_dose* depth_dose);

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
    std::vector<const Rt_depth_dose*> getPeaks();
    /* Weight optimizer */
    void Optimizer();
    void Optimizer2();

};

#endif
