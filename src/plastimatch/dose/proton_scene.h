/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _proton_scene_h_
#define _proton_scene_h_

#include "aperture.h"
#include "plmdose_config.h"
#include "smart_pointer.h"

class Plm_image;
class Proj_matrix;
class Proton_beam;
class Proton_scene_private;
class Volume;

class Rpl_volume;

class PLMDOSE_API Proton_scene {
public:
    SMART_POINTER_SUPPORT (Proton_scene);
public:
    Proton_scene_private *d_ptr;
public:
    Proton_scene ();
    ~Proton_scene ();

    bool init ();

    /* Set the CT volume for dose calculation.
       The Proton_scene takes ownership of this CT. */
    void set_patient (Plm_image*);
    void set_patient (FloatImageType::Pointer&);
    void set_patient (Volume*);
    Volume *get_patient_vol ();
    Plm_image *get_patient ();

    Aperture::Pointer& get_aperture ();

    void set_step_length (double ray_step);

    /* Return the state of the debug flag, which generates debug 
       information on the console */
    bool get_debug () const;
    /* Set the state of the debug flag, which generates debug 
       information on the console */
    void set_debug (bool debug);
    /* Dump state information to the console */
    void debug ();

public:
    Proton_beam *beam;
    Rpl_volume* rpl_vol;
};

#endif
