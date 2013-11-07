/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _ion_plan_h_
#define _ion_plan_h_

#include "plmdose_config.h"
#include "aperture.h"
#include "smart_pointer.h"

class Ion_beam;
class Ion_plan_private;
class Plm_image;
class Proj_matrix;
class Volume;

class Rpl_volume;

class PLMDOSE_API Ion_plan {
public:
    SMART_POINTER_SUPPORT (Ion_plan);
public:
    Ion_plan_private *d_ptr;
public:
    Ion_plan ();
    ~Ion_plan ();

    bool init ();

    /* Set the CT volume for dose calculation.
       The Ion_plan takes ownership of this CT. */
    void set_patient (Plm_image*);
    void set_patient (ShortImageType::Pointer&);
    void set_patient (FloatImageType::Pointer&);
    void set_patient (Volume*);
    Volume *get_patient_vol ();
    Plm_image *get_patient ();

    void set_target (const std::string& target_fn);
    void set_target (UCharImageType::Pointer&);
    void set_target (FloatImageType::Pointer&);
    Plm_image::Pointer& get_target ();

    /* This computes the aperture and range compensator */
    void compute_beam_modifiers ();

    /* This modifies the rpl_volume to account for aperture and 
       range compensator */
    void apply_beam_modifiers ();

    Aperture::Pointer& get_aperture ();
    const Aperture::Pointer& get_aperture () const;

    void set_smearing (float smearing);
    void set_step_length (double ray_step);

    /* Return the state of the debug flag, which generates debug 
       information on the console */
    bool get_debug () const;
    /* Set the state of the debug flag, which generates debug 
       information on the console */
    void set_debug (bool debug);
    /* Dump state information to the console */
    void debug ();

    /* Set beam depth, in mm */
    void set_beam_depth (float z_min, float z_max, float z_step);

    /* Create a dose_volume in the beam frame */
    void dose_volume_create(Volume* dose_volume, float* sigma_max, Rpl_volume* volume);

    /* Compute dose */
    void compute_dose ();
    void compute_dose_push();

    /* Return dose to caller */
    Plm_image::Pointer get_dose ();
    FloatImageType::Pointer get_dose_itk ();

public:
    Ion_beam *beam;
    Rpl_volume* rpl_vol; // contains the radiologic path length along a ray
    Rpl_volume* ct_vol_density; // contains the ct_density along the ray
    Rpl_volume* sigma_vol;  // contains the sigma (lateral spread of the pencil beam - used to calculate the off-axis term) along the ray
};

#endif
