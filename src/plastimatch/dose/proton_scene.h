/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _proton_scene_h_
#define _proton_scene_h_

#include "plmdose_config.h"

class Aperture;
class Plm_image;
class Proj_matrix;
class Proton_Beam;
class Proton_scene_private;
class Volume;

class Rpl_volume;

class PLMDOSE_API Proton_Scene {
public:
    Proton_Scene ();
    ~Proton_Scene ();

    bool init ();
    void print ();
    void set_patient (Plm_image*);
    void set_patient (Volume*);
    void set_step_length (double ray_step);
public:
    Proton_scene_private *d_ptr;
public:
    Aperture* ap;
    Proton_Beam *beam;
    Proj_matrix* pmat;
    Volume* patient;

    Rpl_volume* rpl_vol;
};

#endif
