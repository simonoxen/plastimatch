/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _Rt_plan_h_
#define _Rt_plan_h_

#include "plmdose_config.h"
#include "smart_pointer.h"
#include "plm_image.h"

class Plm_image;
class Proj_matrix;
class Rpl_volume;
class Rt_beam;
class Rt_plan_private;
class Rt_study;
class Volume;

/* Particle type: 0=photon, 1= proton, ions: 2= helium, 3=lithium, 4=beryllium, 5=bore, 6=carbon, 8=oxygen */
enum Particle_type {PARTICLE_TYPE_X=0, PARTICLE_TYPE_P=1, PARTICLE_TYPE_HE=2, PARTICLE_TYPE_LI=3, PARTICLE_TYPE_BE=4, PARTICLE_TYPE_B=5, PARTICLE_TYPE_C=6, PARTICLE_TYPE_O=8};

class PLMDOSE_API Rt_plan {
public:
    SMART_POINTER_SUPPORT (Rt_plan);
public:
    Rt_plan_private *d_ptr;
public:
    Rt_plan ();
    ~Rt_plan ();

    bool init ();

    /* Set the CT volume for dose calculation.
       The Rt_plan takes ownership of this CT/Patient. */

    void set_patient (Plm_image::Pointer&);
    void set_patient (ShortImageType::Pointer&);
    void set_patient (FloatImageType::Pointer&);
    void set_patient (Volume*);

	/* Get the patient volume */

    Volume::Pointer get_patient_volume ();
    Plm_image *get_patient ();

	/* Set/Get the target volume */ 

    void set_target (const std::string& target_fn);
    void set_target (UCharImageType::Pointer&);
    void set_target (FloatImageType::Pointer&);
    Plm_image::Pointer& get_target ();

	/* Set/Get Rt_study */
	void set_rt_study(Rt_study* rt_study);
	Rt_study* get_rt_study();

    /* Return the state of the debug flag, which generates debug 
       information on the console */
    bool get_debug () const;
    /* Set the state of the debug flag, which generates debug 
       information on the console */
    void set_debug (bool debug);
    /* Dump state information to the console */
    void debug ();

	/* Set source size in mm */
    void set_normalization_dose(float normalization_dose);

    /* Get source size in mm */
    float get_normalization_dose();

    /* Compute dose */
    void compute_dose ();

    /* Get outputs */
    Plm_image::Pointer get_dose ();
    FloatImageType::Pointer get_dose_itk ();
	void set_dose(Plm_image::Pointer& dose);

public:

    Rt_beam *beam;
	std::vector<Rt_beam*> beam_storage;
};

static inline void display_progress (float is, float of);

#endif
