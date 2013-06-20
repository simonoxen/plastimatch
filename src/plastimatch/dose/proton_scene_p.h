/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _proton_scene_p_h_
#define _proton_scene_p_h_

#include "plmdose_config.h"

#include "aperture.h"
#include "plm_image.h"

class PLMDOSE_API Proton_scene_private {
public:
    Proton_scene_private ();
    ~Proton_scene_private ();
public:
    bool debug;
    double step_length;
    Plm_image::Pointer patient;
    Plm_image::Pointer target;
    Plm_image::Pointer dose;
    Aperture::Pointer ap;

    float z_min;
    float z_max;
    float z_step;
};

#endif
