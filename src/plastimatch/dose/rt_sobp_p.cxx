/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plmdose_config.h"

#include "logfile.h"
#include "rt_depth_dose.h"
#include "rt_sobp_p.h"
#include "rt_lut.h"

Rt_sobp_private::Rt_sobp_private ()
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

Rt_sobp_private::Rt_sobp_private (Particle_type particle_type)
{
    d_lut = new float[0];
    e_lut = new float[0];
    dres = 1.0;
	dose_max = 1.f;
    num_samples = 0;
    eres = 2.0;
    E_min = 0;
    E_max = 0;
    dmin = 0.0;
    dmax = 0.0;
    dend = 0.0;
    prescription_dmin = 50.f;
    prescription_dmax = 100.f;
    set_particle_type (particle_type);
}

Rt_sobp_private::Rt_sobp_private (const Rt_sobp_private* rsp)
{
    d_lut = new float[0];
    e_lut = new float[0];
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

Rt_sobp_private::~Rt_sobp_private ()
{
    if (d_lut) delete[] d_lut;
    if (e_lut) delete[] e_lut;
	if (f_lut) delete[] f_lut;
    clear_peaks ();
}

void
Rt_sobp_private::set_particle_type (Particle_type particle_type)
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

void
Rt_sobp_private::clear_peaks ()
{
    std::vector<const Rt_depth_dose*>::iterator it;
    for (it = depth_dose.begin(); it != depth_dose.end(); ++it) {
        delete *it;
    }
    depth_dose.clear();
    sobp_weight.clear();
}
