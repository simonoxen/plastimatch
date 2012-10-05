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
class Volume;

typedef struct rpl_volume Rpl_volume;

class PLMDOSE_API Proton_Scene {
public:
    Proton_Scene ();
    ~Proton_Scene ();

    bool init (int ray_step);
    void set_patient (Plm_image*);
    void set_patient (Volume*);
    void print ();
public:
    Aperture* ap;
    Proton_Beam *beam;
    Proj_matrix* pmat;
    Volume* patient;

    Rpl_volume* rpl_vol;
};

#endif
