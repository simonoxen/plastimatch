/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _photon_plan_p_h_
#define _photon_plan_p_h_

#include "plmdose_config.h"

#include "aperture.h"
#include "plm_image.h"

class PLMDOSE_API Photon_plan_private {
public:
    Photon_plan_private ();
    ~Photon_plan_private ();
public:
    bool debug;
    double step_length;
    float smearing;
    float source_size;

    Plm_image::Pointer patient;
    Plm_image::Pointer target;
    Plm_image::Pointer dose;
    Aperture::Pointer ap;

    float z_min;
    float z_max;
    float z_step;
};

#endif
