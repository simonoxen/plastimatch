/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plmdose_config.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "aperture.h"
#include "plm_image.h"
#include "proton_beam.h"
#include "proj_matrix.h"
#include "proton_scene.h"
#include "rpl_volume.h"
#include "volume.h"

class Proton_scene_private {
public:
    Proton_scene_private () {
        debug = false;
        step_length = 0.;
        patient = 0;
    }
    ~Proton_scene_private () {
        if (patient) {
            delete patient;
        }
    }
public:
    bool debug;
    double step_length;
    Plm_image *patient;
};

Proton_Scene::Proton_Scene ()
{
    this->d_ptr = new Proton_scene_private;
    this->ap = new Aperture;
    this->beam = new Proton_Beam;
    this->pmat = new Proj_matrix;

    this->rpl_vol = 0;
}

Proton_Scene::~Proton_Scene ()
{
    delete this->d_ptr;
    delete this->ap;
    delete this->beam;
    delete this->pmat;
    if (this->rpl_vol) {
        delete this->rpl_vol;
    }
}

void
Proton_Scene::set_step_length (double step_length)
{
    d_ptr->step_length = step_length;
}

bool
Proton_Scene::init ()
{
    if (!this->ap) return false;
    if (!this->beam) return false;
    if (!this->get_patient()) return false;

    this->rpl_vol = new Rpl_volume;
    this->rpl_vol->set_geometry (
        this->beam->get_source_position(),
        this->beam->get_isocenter_position(),
        this->ap->vup,
        this->ap->get_distance(),
        this->ap->get_dim(),
        this->ap->get_center(),
        this->ap->get_spacing(),
        d_ptr->step_length);
        
    if (!this->rpl_vol) return false;

    /* scan through aperture to fill in rpl_volume */
    this->rpl_vol->compute (d_ptr->patient->get_volume_float_raw());

    return true;
}

void
Proton_Scene::set_patient (Plm_image* ct_vol)
{
    d_ptr->patient = ct_vol;
}

void
Proton_Scene::set_patient (Volume* ct_vol)
{
    d_ptr->patient = new Plm_image;
    d_ptr->patient->set_volume (ct_vol);
}

Volume *
Proton_Scene::get_patient_vol ()
{
    return d_ptr->patient->get_volume_float_raw ();
}

Plm_image *
Proton_Scene::get_patient ()
{
    return d_ptr->patient;
}

bool
Proton_Scene::get_debug (void) const
{
    return d_ptr->debug;
}

void
Proton_Scene::set_debug (bool debug)
{
    d_ptr->debug = debug;
}

void
Proton_Scene::debug ()
{
    Aperture* ap = this->ap;
    Proton_Beam* beam = this->beam;

    printf ("BEAM\n");
    printf ("  -- [POS] Source :   %g %g %g\n", 
        beam->get_source_position(0), beam->get_source_position(1), 
        beam->get_source_position(2));
    printf ("  -- [POS] Isocenter: %g %g %g\n", 
        beam->get_isocenter_position(0), beam->get_isocenter_position(1), 
        beam->get_isocenter_position(2));
    printf ("APERTURE\n");
    printf ("  -- [NUM] Res   : %i %i\n", ap->get_dim(0), ap->get_dim(1));
    printf ("  -- [DIS] Offset: %g\n", ap->get_distance());
    printf ("  -- [POS] Center: %g %g %g\n", ap->ic_room[0], ap->ic_room[1], ap->ic_room[2]);
    printf ("  -- [POS] UpLeft: %g %g %g\n", ap->ul_room[0], ap->ul_room[1], ap->ul_room[2]);
    printf ("  -- [VEC] Up    : %g %g %g\n", ap->vup[0], ap->vup[1], ap->vup[2]);
    printf ("  -- [VEC] Normal: %g %g %g\n", ap->nrm[0], ap->nrm[1], ap->nrm[2]);
    printf ("  -- [VEC] Right : %g %g %g\n", ap->prt[0], ap->prt[1], ap->prt[2]);
    printf ("  -- [VEC] Down  : %g %g %g\n", ap->pdn[0], ap->pdn[1], ap->pdn[2]);
    printf ("  -- [VEC] col++ : %g %g %g\n", ap->incr_c[0], ap->incr_c[1], ap->incr_c[2]);
    printf ("  -- [VEC] row++ : %g %g %g\n", ap->incr_r[0], ap->incr_r[1], ap->incr_r[2]);
}
