/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plmdose_config.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "aperture.h"
#include "ion_beam.h"
#include "ion_dose.h"
#include "ion_plan.h"
#include "ion_plan_p.h"
#include "ion_sobp.h"
#include "plm_image.h"
#include "proj_matrix.h"
#include "rpl_volume.h"
#include "volume.h"
#include "volume_macros.h"

Ion_plan::Ion_plan ()
{
    printf ("*** Creating proton scene ***\n");
    this->d_ptr = new Ion_plan_private;
    this->beam = new Ion_beam;
    this->rpl_vol = 0;
}

Ion_plan::~Ion_plan ()
{
    delete this->d_ptr;
    delete this->beam;
    if (this->rpl_vol) {
        delete this->rpl_vol;
    }
}

void
Ion_plan::set_step_length (double step_length)
{
    d_ptr->step_length = step_length;
}

bool
Ion_plan::init ()
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
    this->rpl_vol->set_ct_volume (d_ptr->patient);
    this->rpl_vol->compute_rpl ();

    return true;
}

void
Ion_plan::set_patient (Plm_image* ct_vol)
{
    d_ptr->patient.reset(ct_vol);
}

void
Ion_plan::set_patient (ShortImageType::Pointer& ct_vol)
{
    d_ptr->patient->set_itk (ct_vol);

    /* compute_segdepth_volume assumes float */
    d_ptr->patient->convert (PLM_IMG_TYPE_GPUIT_FLOAT);
}

void
Ion_plan::set_patient (FloatImageType::Pointer& ct_vol)
{
    d_ptr->patient->set_itk (ct_vol);
}

void
Ion_plan::set_patient (Volume* ct_vol)
{
    d_ptr->patient->set_volume (ct_vol);
}

Volume *
Ion_plan::get_patient_vol ()
{
    return d_ptr->patient->get_vol_float ();
}

Plm_image *
Ion_plan::get_patient ()
{
    return d_ptr->patient.get();
}

void
Ion_plan::set_target (const std::string& target_fn)
{
    d_ptr->target = Plm_image::New (new Plm_image (target_fn));

    /* compute_segdepth_volume assumes float */
    d_ptr->target->convert (PLM_IMG_TYPE_GPUIT_FLOAT);
}

void
Ion_plan::set_target (UCharImageType::Pointer& target_vol)
{
    d_ptr->target->set_itk (target_vol);

    /* compute_segdepth_volume assumes float */
    d_ptr->target->convert (PLM_IMG_TYPE_GPUIT_FLOAT);
}

void
Ion_plan::set_target (FloatImageType::Pointer& target_vol)
{
    d_ptr->target->set_itk (target_vol);
}

Plm_image::Pointer&
Ion_plan::get_target ()
{
    return d_ptr->target;
}

void
Ion_plan::compute_beam_modifiers ()
{
    this->rpl_vol->compute_beam_modifiers (
        d_ptr->target->get_vol(), 0);
}

void
Ion_plan::apply_beam_modifiers ()
{
    this->rpl_vol->apply_beam_modifiers ();
}

Aperture::Pointer&
Ion_plan::get_aperture () 
{
    return d_ptr->ap;
}

const Aperture::Pointer&
Ion_plan::get_aperture () const
{
    return d_ptr->ap;
}

bool
Ion_plan::get_debug (void) const
{
    return d_ptr->debug;
}

void
Ion_plan::set_debug (bool debug)
{
    d_ptr->debug = debug;
}

void
Ion_plan::set_beam_depth (float z_min, float z_max, float z_step)
{
    d_ptr->z_min = z_min;
    d_ptr->z_max = z_max;
    d_ptr->z_step = z_step;
}

Plm_image::Pointer
Ion_plan::get_dose ()
{
    return d_ptr->dose;
}

FloatImageType::Pointer
Ion_plan::get_dose_itk ()
{
    return d_ptr->dose->itk_float();
}

void
Ion_plan::debug ()
{
    Aperture::Pointer& ap = d_ptr->ap;
    Ion_beam* beam = this->beam;

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

static inline void
display_progress (
    float is,
    float of
) 
{
#if defined (PROGRESS)
    printf (" [%3i%%]\b\b\b\b\b\b\b",
           (int)floorf((is/of)*100.0f));
    fflush (stdout);
#endif
}

void
Ion_plan::compute_dose ()
{
    Ion_beam* beam = this->beam;
    Volume* ct_vol = this->get_patient_vol ();
    Rpl_volume* rpl_vol = this->rpl_vol;

    Volume* dose_vol = volume_clone_empty (ct_vol);
    float* dose_img = (float*) dose_vol->img;

    if (this->get_debug()) {
        rpl_vol->save ("beam_debug/depth_vol.mha");
        beam->dump ("beam_debug");
    }

    /* scan through patient CT Volume */
    plm_long ct_ijk[3];
    double ct_xyz[4];
    plm_long idx = 0;
    for (ct_ijk[2] = 0; ct_ijk[2] < ct_vol->dim[2]; ct_ijk[2]++) {
        for (ct_ijk[1] = 0; ct_ijk[1] < ct_vol->dim[1]; ct_ijk[1]++) {
            for (ct_ijk[0] = 0; ct_ijk[0] < ct_vol->dim[0]; ct_ijk[0]++) {
                double dose = 0.0;

                bool voxel_debug = false;
#if defined (commentout)
                if (ct_ijk[2] == 60 && ct_ijk[1] == 44 && ct_ijk[0] == 5) {
                    voxel_debug = true;
                }
#endif

                /* Transform vol index into space coords */
                ct_xyz[0] = (double) (ct_vol->offset[0] + ct_ijk[0] * ct_vol->spacing[0]);
                ct_xyz[1] = (double) (ct_vol->offset[1] + ct_ijk[1] * ct_vol->spacing[1]);
                ct_xyz[2] = (double) (ct_vol->offset[2] + ct_ijk[2] * ct_vol->spacing[2]);
                ct_xyz[3] = (double) 1.0;

                if (voxel_debug) {
                    printf ("Voxel (%d, %d, %d) -> (%f, %f, %f)\n",
                        (int) ct_ijk[0], (int) ct_ijk[1], (int) ct_ijk[2], 
                        ct_xyz[0], ct_xyz[1], ct_xyz[2]);
                }

                switch (beam->get_flavor()) {
                case 'a':
                    dose = dose_direct (ct_xyz, this);
                    break;
                case 'b':
                    dose = dose_scatter (ct_xyz, ct_ijk, this);
                    break;
                case 'c':
                    dose = dose_hong (ct_xyz, ct_ijk, this);
                    break;
                case 'd':
                    dose = dose_debug (ct_xyz, this);
                    break;
                }

                /* Insert the dose into the dose volume */
                idx = volume_index (dose_vol->dim, ct_ijk);
                dose_img[idx] = dose;
            }
        }
        display_progress ((float)idx, (float)ct_vol->npix);
    }

    Plm_image::Pointer dose = Plm_image::New();
    dose->set_volume (dose_vol);
    d_ptr->dose = dose;
}
