/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _rt_plan_h_
#define _rt_plan_h_

#include "plmdose_config.h"
#include "plm_image.h"
#include "plm_return_code.h"
#include "smart_pointer.h"
#include "threading.h"

class Plm_image;
class Proj_matrix;
class Rpl_volume;
class Rt_beam;
class Rt_plan_private;
class Rt_study;
class Volume;

class PLMDOSE_API Rt_plan {
public:
    SMART_POINTER_SUPPORT (Rt_plan);
public:
    Rt_plan_private *d_ptr;
public:
    Rt_plan ();
    ~Rt_plan ();

public:
    Plm_return_code parse_args (int argc, char* argv[]);

    bool init ();

    /* Set the CT volume for dose calculation.
       The Rt_plan takes ownership of this CT/Patient. */
    void set_patient (const std::string& patient_fn);
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

    /* Get/Set Rt_beam(s) */
    Rt_beam* append_beam ();
    Rt_beam* get_last_rt_beam ();

    /* Return the state of the debug flag, which generates debug 
       information on the console */
    bool get_debug () const;
    /* Set the state of the debug flag, which generates debug 
       information on the console */
    void set_debug (bool debug);

    /* Set/Get threading */ 
    void set_threading (Threading threading);

    /* Set/Set normalization dose */
    void set_normalization_dose (float normalization_dose);
    float get_normalization_dose ();

    /* Compute dose */
    Plm_return_code compute_plan ();
    void compute_dose ();

    /* Get outputs */
    Plm_image::Pointer get_dose ();
    FloatImageType::Pointer get_dose_itk ();
    void set_output_dose (const std::string& output_dose_fn);
    void set_dose(Plm_image::Pointer& dose);

    void print_verif ();

public:

    Rt_beam *beam;
    std::vector<Rt_beam*> beam_storage;
};

#endif
