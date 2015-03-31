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
#ifndef _rt_sobp_p_h_
#define _rt_sobp_p_h_

#include "plmdose_config.h"
#include <vector>
#include "particle_type.h"

class Rt_depth_dose;

class Rt_sobp_private {
public:
    std::vector<const Rt_depth_dose*> depth_dose;

    float* d_lut;               /* depth array (mm) */
    float* e_lut;               /* energy array (MeV) */
	float* f_lut;				/* integrated energy array (MeV) */
    double dres;
	float dose_max;
    int num_samples;	        /* number of depths */

    int eres;			/* energy resolution */
    int num_peaks;		/* number of peaks */

    std::vector<double> sobp_weight;

    int E_min;			/* lower energy */
    int E_max;			/* higher energy */

    float dmin;			/* lower depth */
    float dmax;			/* higher depth */
    float dend;			/* end of the depth array */

    /* p  & alpha are parameters that bind depth and energy 
       according to ICRU */
    Particle_type particle_type;
    double p;			
    double alpha;

    float prescription_dmin;
    float prescription_dmax;

public:
    Rt_sobp_private ();
    Rt_sobp_private (Particle_type);
    Rt_sobp_private (const Rt_sobp_private*);
    ~Rt_sobp_private ();
public:
    void set_particle_type (Particle_type);
    void clear_peaks ();
};

#endif
