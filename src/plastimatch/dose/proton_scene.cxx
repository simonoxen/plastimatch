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
#include "proton_dose.h"
#include "proton_scene.h"
#include "proton_scene_p.h"
#include "rpl_volume.h"
#include "sobp.h"
#include "volume.h"

Proton_scene::Proton_scene ()
{
    printf ("*** Creating proton scene ***\n");
    this->d_ptr = new Proton_scene_private;
    this->beam = new Proton_beam;
    this->rpl_vol = 0;
}

Proton_scene::~Proton_scene ()
{
    delete this->d_ptr;
    delete this->beam;
    if (this->rpl_vol) {
        delete this->rpl_vol;
    }
}

void
Proton_scene::set_step_length (double step_length)
{
    d_ptr->step_length = step_length;
}

bool
Proton_scene::init ()
{
    if (!this->beam) return false;
    if (!this->get_patient()) return false;

    this->rpl_vol = new Rpl_volume;
    this->rpl_vol->set_geometry (
        this->beam->get_source_position(),
        this->beam->get_isocenter_position(),
        d_ptr->ap->vup,
        d_ptr->ap->get_distance(),
        d_ptr->ap->get_dim(),
        d_ptr->ap->get_center(),
        d_ptr->ap->get_spacing(),
        d_ptr->step_length);
        
    if (!this->rpl_vol) return false;

    /* Copy aperture from scene into rpl volume */
    this->rpl_vol->set_aperture (d_ptr->ap);

    /* Scan through aperture to fill in rpl_volume */
    this->rpl_vol->compute (d_ptr->patient->get_volume_float_raw());

    return true;
}

void
Proton_scene::set_patient (Plm_image* ct_vol)
{
    d_ptr->patient.reset(ct_vol);
}

void
Proton_scene::set_patient (ShortImageType::Pointer& ct_vol)
{
    d_ptr->patient->set_itk (ct_vol);

    /* compute_segdepth_volume assumes float */
    d_ptr->patient->convert (PLM_IMG_TYPE_GPUIT_FLOAT);
}

void
Proton_scene::set_patient (FloatImageType::Pointer& ct_vol)
{
    d_ptr->patient->set_itk (ct_vol);
}

void
Proton_scene::set_patient (Volume* ct_vol)
{
    d_ptr->patient->set_volume (ct_vol);
}

Volume *
Proton_scene::get_patient_vol ()
{
    return d_ptr->patient->get_volume_float_raw ();
}

Plm_image *
Proton_scene::get_patient ()
{
    return d_ptr->patient.get();
}

void
Proton_scene::set_target (UCharImageType::Pointer& target_vol)
{
    d_ptr->target->set_itk (target_vol);

    /* compute_segdepth_volume assumes float */
    d_ptr->target->convert (PLM_IMG_TYPE_GPUIT_FLOAT);
}

void
Proton_scene::set_target (FloatImageType::Pointer& target_vol)
{
    d_ptr->target->set_itk (target_vol);
}

Plm_image::Pointer&
Proton_scene::get_target ()
{
    return d_ptr->target;
}

void
Proton_scene::compute_beam_modifiers ()
{
    this->rpl_vol->compute_segdepth_volume (
        d_ptr->target->get_volume(), 0);
}

Aperture::Pointer&
Proton_scene::get_aperture () 
{
    return d_ptr->ap;
}

const Aperture::Pointer&
Proton_scene::get_aperture () const
{
    return d_ptr->ap;
}

bool
Proton_scene::get_debug (void) const
{
    return d_ptr->debug;
}

void
Proton_scene::set_debug (bool debug)
{
    d_ptr->debug = debug;
}

void
Proton_scene::set_beam_depth (float z_min, float z_max, float z_step)
{
    d_ptr->z_min = z_min;
    d_ptr->z_max = z_max;
    d_ptr->z_step = z_step;

    Sobp sobp;
}

Plm_image::Pointer
Proton_scene::get_dose ()
{
    return d_ptr->dose;
}

void
Proton_scene::debug ()
{
    Aperture::Pointer& ap = d_ptr->ap;
    Proton_beam* beam = this->beam;

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
