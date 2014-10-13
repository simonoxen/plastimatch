/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plmdose_config.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "aperture.h"
#include "dose_volume_functions.h"
#include "ion_beam.h"
#include "ion_dose.h"
#include "ion_plan.h"
#include "ion_plan_p.h"
#include "ion_pristine_peak.h"
#include "ion_sigma.h"
#include "ion_sobp.h"
#include "plm_image.h"
#include "plm_timer.h"
#include "proj_matrix.h"
#include "proj_volume.h"
#include "radiation_lut.h"
#include "ray_data.h"
#include "rpl_volume.h"
#include "volume.h"
#include "volume_macros.h"

Ion_plan::Ion_plan ()
{
    printf ("*** Creating proton scene ***\n");
    this->d_ptr = new Ion_plan_private;
    this->beam = new Ion_beam;
    this->rpl_vol = 0;
    if (this->beam->get_flavor() == 'f')
    {
        this->ct_vol_density = 0;
        this->sigma_vol = 0;
    }
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
Ion_plan::set_smearing (float smearing)
{
    d_ptr->smearing = smearing;
}

void
Ion_plan::set_step_length (double step_length)
{
    d_ptr->step_length = step_length;
}

double 
Ion_plan::get_step_length()
{
    return d_ptr->step_length;
}

bool
Ion_plan::init ()
{
    if (!this->beam) return false;
    if (!this->get_patient()) return false;

	this->aperture_vol = new Rpl_volume;

    if (!this->rpl_vol) {this->rpl_vol = new Rpl_volume;}
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

    if (this->beam->get_flavor() == 'f'|| this->beam->get_flavor() == 'g' || this->beam->get_flavor() == 'h')
    {
        /* building the ct_density_vol */
        this->ct_vol_density = new Rpl_volume;
        this->ct_vol_density->set_geometry (
            this->beam->get_source_position(),
            this->beam->get_isocenter_position(),
            d_ptr->ap->vup,
            d_ptr->ap->get_distance(),
            d_ptr->ap->get_dim(),
            d_ptr->ap->get_center(),
            d_ptr->ap->get_spacing(),
            d_ptr->step_length);        
        if (!this->ct_vol_density) return false;

        /* building the sigma_vol */
        this->sigma_vol = new Rpl_volume;
        this->sigma_vol->set_geometry (
            this->beam->get_source_position(),
            this->beam->get_isocenter_position(),
            d_ptr->ap->vup,
            d_ptr->ap->get_distance(),
            d_ptr->ap->get_dim(),
            d_ptr->ap->get_center(),
            d_ptr->ap->get_spacing(),
            d_ptr->step_length);
        
        if (!this->sigma_vol) return false;
    }

    /* Copy aperture from scene into rpl volume */
    this->rpl_vol->set_aperture (d_ptr->ap);

    if (this->beam->get_flavor() == 'f' || this->beam->get_flavor() == 'g' || this->beam->get_flavor() == 'h')
    {
        this->ct_vol_density->set_aperture (d_ptr->ap);
        this->sigma_vol->set_aperture (d_ptr->ap);
    }

    /* Scan through aperture to fill in rpl_volume */
    this->rpl_vol->set_ct_volume (d_ptr->patient);

    if (this->beam->get_flavor() == 'f' || this->beam->get_flavor() == 'g' || this->beam->get_flavor() == 'h')
    {
        if(this->rpl_vol->get_ct() && this->rpl_vol->get_ct_limit())
        {
            /* We don't do everything again, we just copy the ct & ct_limits as all the volumes geometrically equal*/
            this->ct_vol_density->set_ct (this->rpl_vol->get_ct());
            this->ct_vol_density->set_ct_limit(this->rpl_vol->get_ct_limit());
        
            this->sigma_vol->set_ct(this->rpl_vol->get_ct());
            this->sigma_vol->set_ct_limit(this->rpl_vol->get_ct_limit());
        }
        else
        {
            printf("ray_data or clipping planes to be copied from rpl volume don't exist\n");
        }
    }
    
    /*Now we can compute the rpl_volume*/
    this->rpl_vol->compute_rpl ();
    
    /* and the others */
    if (this->beam->get_flavor() == 'f' || this->beam->get_flavor() == 'g' || this->beam->get_flavor() == 'h')
    {
        if(this->rpl_vol->get_Ray_data() && this->rpl_vol->get_front_clipping_plane() && this->rpl_vol->get_back_clipping_plane())
        {
            /* We don't do everything again, we just copy the ray_data & clipping planes as all the volumes geometrically equal*/
            this->ct_vol_density->set_ray(this->rpl_vol->get_Ray_data());
            this->ct_vol_density->set_front_clipping_plane(this->rpl_vol->get_front_clipping_plane());
            this->ct_vol_density->set_back_clipping_plane(this->rpl_vol->get_back_clipping_plane());
        
            this->sigma_vol->set_ray(this->rpl_vol->get_Ray_data());
            this->sigma_vol->set_front_clipping_plane(this->rpl_vol->get_front_clipping_plane());
            this->sigma_vol->set_back_clipping_plane(this->rpl_vol->get_back_clipping_plane());
        }
        else
        {
            printf("ct or ct_limits to be copied from rpl_vol don't exist\n");
        }
    }
    return true;
}

void
Ion_plan::set_patient (Plm_image::Pointer& ct_vol)
{
    d_ptr->patient = ct_vol;
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

Volume::Pointer
Ion_plan::get_patient_volume ()
{
    return d_ptr->patient->get_volume_float ();
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
    /* Compute the aperture and compensator */
    this->rpl_vol->compute_beam_modifiers (
        d_ptr->target->get_vol(), 0);

    /* Apply smearing */
    d_ptr->ap->apply_smearing (d_ptr->smearing);
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
Ion_plan::set_normalization_dose(float normalization_dose)
{
	d_ptr->normalization_dose = normalization_dose;
}

float
Ion_plan::get_normalization_dose()
{
	return d_ptr->normalization_dose;
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
    printf ("-- compute_dose entry --\n");
    Ion_beam* beam = this->beam;
    Volume::Pointer ct_vol = this->get_patient_volume ();
    Volume::Pointer dose_vol = ct_vol->clone_empty ();
    float* dose_img = (float*) dose_vol->img;

    Volume* dose_volume_tmp = new Volume;
    float* dose_img_tmp = (float*) dose_volume_tmp->img;

    UNUSED_VARIABLE (dose_img_tmp);

    float margin = 0;
	int margins[2] = {0,0};
    double range = 0;
    int new_dim[2]={0,0};
    double new_center[2]={0,0};
    double biggest_sigma_ever = 0;
    Plm_timer timer;
    double time_sigma_conv = 0.0;
    double time_dose_calc = 0.0;
    double time_dose_misc = 0.0;
    double time_dose_reformat = 0.0;

    if (this->beam->get_flavor() == 'f' || this->beam->get_flavor() == 'g' || this->beam->get_flavor() == 'h')
    {
        float sigmaMax = 0;
        float *sigma_max =&sigmaMax; // used to find the max sigma in the volume and add extra margins during the dose creation volume

        printf ("Computing rpl_ct\n");
        this->ct_vol_density->compute_rpl_ct ();

        printf ("Computing_void_rpl\n");
        this->sigma_vol->compute_rpl_rglength_wo_rg_compensator(); // we compute the rglength in the sigma_volume, without the range compensator as it will be added by a different process

        Rpl_volume* rpl_vol = this->rpl_vol;
        Rpl_volume* sigma_vol = this->sigma_vol;

        float* sigma_img = (float*) sigma_vol->get_vol()->img;

        /* building the sigma_dose_vol */
        if (this->beam->get_flavor() == 'g') {
            this->rpl_dose_vol = new Rpl_volume;
        }

        if (this->beam->get_flavor() == 'h') {
            this->rpl_vol_lg = new Rpl_volume;
            this->ct_vol_density_lg = new Rpl_volume;
            this->sigma_vol_lg = new Rpl_volume;
        }

        printf ("More setup\n");
        std::vector<const Ion_pristine_peak*> peaks = this->beam->get_sobp()->getPeaks();

        std::vector<const Ion_pristine_peak*>::const_reverse_iterator it;
        for (it = peaks.rbegin (); it <peaks.rend(); it++) {
            const Ion_pristine_peak *ppp = *it;
            printf("Building dose matrix for %lg MeV beamlets - \n", ppp->E0);
            timer.start ();
			compute_sigmas(this, ppp->E0, sigma_max, "small", margins);
            time_sigma_conv += timer.report ();

            if (this->beam->get_flavor() == 'f') // Desplanques' algorithm
            {
                range = 10 * getrange(ppp->E0); // range in mm
                dose_volume_create(dose_volume_tmp, sigma_max, this->rpl_vol, range);
                compute_dose_ray_desplanques(dose_volume_tmp, ct_vol, rpl_vol, sigma_vol, ct_vol_density, this->beam, dose_vol, ppp, this->get_normalization_dose());
            }
            else if (this->beam->get_flavor() == 'g') // Sharp's algorithm
            {
                timer.start ();
                if (*sigma_max > biggest_sigma_ever)
                {
                    biggest_sigma_ever = *sigma_max;
                    /* Calculating the pixel-margins of the aperture to take into account the scattering*/
                    margin = (float) 3 * (*sigma_max)/(this->get_aperture()->get_distance()+this->rpl_vol->get_front_clipping_plane()) * this->get_aperture()->get_distance()+1;
                    margins[0] = ceil (margin/vec3_len(this->rpl_vol->get_proj_volume()->get_incr_c()));
                    margins[1] = ceil (margin/vec3_len(this->rpl_vol->get_proj_volume()->get_incr_r()));
                    new_dim[0] = this->rpl_vol->get_aperture()->get_dim(0) + 2 * margins[0];
                    new_dim[1] = this->rpl_vol->get_aperture()->get_dim(1) + 2 * margins[1];
                    new_center[0] = this->rpl_vol->get_aperture()->get_center(0) + margins[0];
                    new_center[1] = this->rpl_vol->get_aperture()->get_center(1) + margins[1];

                    this->rpl_dose_vol->get_aperture()->set_center(new_center);
                    this->rpl_dose_vol->get_aperture()->set_dim(new_dim);

                    this->rpl_dose_vol->get_aperture()->set_distance(this->rpl_vol->get_aperture()->get_distance());
                    this->rpl_dose_vol->get_aperture()->set_spacing(this->rpl_vol->get_aperture()->get_spacing());

                    this->rpl_dose_vol->set_geometry (
                        this->beam->get_source_position(),
                        this->beam->get_isocenter_position(),
                        d_ptr->ap->vup,
                        d_ptr->ap->get_distance(),
                        this->rpl_dose_vol->get_aperture()->get_dim(),
                        this->rpl_dose_vol->get_aperture()->get_center(),
                        d_ptr->ap->get_spacing(),
                        d_ptr->step_length);

                    this->rpl_dose_vol->set_ct(this->rpl_vol->get_ct());
                    this->rpl_dose_vol->set_ct_limit(this->rpl_vol->get_ct_limit());
                    this->rpl_dose_vol->compute_ray_data();
				
                    this->rpl_dose_vol->set_front_clipping_plane(this->rpl_vol->get_front_clipping_plane());
                    this->rpl_dose_vol->set_back_clipping_plane(this->rpl_vol->get_back_clipping_plane());
                }
                /* update the dose_vol with the CT values before to calculate the dose */
                this->rpl_dose_vol->compute_rpl_void();
                time_dose_misc += timer.report ();

                /* dose calculation in the rpl_dose_volume */
                timer.start ();
                compute_dose_ray_sharp (ct_vol, rpl_vol, sigma_vol, 
                    ct_vol_density, this->beam, rpl_dose_vol, d_ptr->ap, 
                    ppp, margins, this->get_normalization_dose());
                time_dose_calc += timer.report ();

                timer.start ();
                dose_volume_reconstruction(rpl_dose_vol, dose_vol);
                time_dose_reformat += timer.report ();
            }

            if (this->beam->get_flavor() == 'h') // Shackleford's algorithm
            {

                /* Calculating the pixel-margins of the aperture to take into account the scattering*/
                margin = (float) 3 * (*sigma_max)/(this->get_aperture()->get_distance()+this->rpl_vol->get_front_clipping_plane()) * this->get_aperture()->get_distance()+1;
                margins[0] = ceil (margin/vec3_len(this->rpl_vol->get_proj_volume()->get_incr_c()));
                margins[1] = ceil (margin/vec3_len(this->rpl_vol->get_proj_volume()->get_incr_r()));
                new_dim[0] = this->rpl_vol->get_aperture()->get_dim(0) + 2 * margins[0];
                new_dim[1] = this->rpl_vol->get_aperture()->get_dim(1) + 2 * margins[1];
                new_center[0] = this->rpl_vol->get_aperture()->get_center(0) + margins[0];
                new_center[1] = this->rpl_vol->get_aperture()->get_center(1) + margins[1];

                int radius_sample = 4;
                int theta_sample = 8;
                std::vector<double> xy_grid (2*(radius_sample * theta_sample),0); // contains the xy coordinates of the sectors in the plane; the central pixel is not included in this vector. 
                std::vector<double> area (radius_sample, 0); // contains the areas of the sectors

                this->rpl_vol_lg->get_aperture()->set_center(new_center);
                this->rpl_vol_lg->get_aperture()->set_dim(new_dim);
                this->rpl_vol_lg->get_aperture()->set_distance(this->rpl_vol->get_aperture()->get_distance());
                this->rpl_vol_lg->get_aperture()->set_spacing(this->rpl_vol->get_aperture()->get_spacing());
                this->rpl_vol_lg->set_geometry (this->beam->get_source_position(), this->beam->get_isocenter_position(), this->get_aperture()->vup, this->get_aperture()->get_distance(), this->rpl_vol_lg->get_aperture()->get_dim(), this->rpl_vol_lg->get_aperture()->get_center(), this->get_aperture()->get_spacing(), this->get_step_length());
                this->rpl_vol_lg->set_ct(this->rpl_vol->get_ct());
                this->rpl_vol_lg->set_ct_limit(this->rpl_vol->get_ct_limit());
                this->rpl_vol_lg->compute_ray_data();
				this->rpl_vol_lg->compute_rpl();

                this->ct_vol_density_lg->get_aperture()->set_center(new_center);
                this->ct_vol_density_lg->get_aperture()->set_dim(new_dim);
                this->ct_vol_density_lg->get_aperture()->set_distance(this->rpl_vol->get_aperture()->get_distance());
                this->ct_vol_density_lg->get_aperture()->set_spacing(this->rpl_vol->get_aperture()->get_spacing());
                this->ct_vol_density_lg->set_geometry (this->beam->get_source_position(), this->beam->get_isocenter_position(), this->get_aperture()->vup, this->get_aperture()->get_distance(), this->rpl_vol_lg->get_aperture()->get_dim(), this->rpl_vol_lg->get_aperture()->get_center(), this->get_aperture()->get_spacing(), this->get_step_length());
                this->ct_vol_density_lg->set_ct(this->rpl_vol->get_ct());
                this->ct_vol_density_lg->set_ct_limit(this->rpl_vol->get_ct_limit());
                this->ct_vol_density_lg->compute_ray_data();
                this->rpl_vol_lg->set_front_clipping_plane(this->rpl_vol_lg->get_front_clipping_plane());
                this->rpl_vol_lg->set_back_clipping_plane(this->rpl_vol_lg->get_back_clipping_plane());
                this->ct_vol_density_lg->compute_rpl_ct();

                this->sigma_vol_lg->get_aperture()->set_center(new_center);
                this->sigma_vol_lg->get_aperture()->set_dim(new_dim);
				
                this->sigma_vol_lg->get_aperture()->set_distance(this->rpl_vol->get_aperture()->get_distance());
                this->sigma_vol_lg->get_aperture()->set_spacing(this->rpl_vol->get_aperture()->get_spacing());
                this->sigma_vol_lg->set_geometry (this->beam->get_source_position(), this->beam->get_isocenter_position(), this->get_aperture()->vup, this->get_aperture()->get_distance(), this->rpl_vol_lg->get_aperture()->get_dim(), this->rpl_vol_lg->get_aperture()->get_center(), this->get_aperture()->get_spacing(), this->get_step_length());
                this->sigma_vol_lg->set_ct(this->rpl_vol->get_ct());
                this->sigma_vol_lg->set_ct_limit(this->rpl_vol->get_ct_limit());
                this->sigma_vol_lg->compute_ray_data();
                this->sigma_vol_lg->set_front_clipping_plane(this->rpl_vol_lg->get_front_clipping_plane());
                this->sigma_vol_lg->set_back_clipping_plane(this->rpl_vol_lg->get_back_clipping_plane());
                this->sigma_vol_lg->compute_rpl_rglength_wo_rg_compensator();

				if (this->get_aperture()->have_aperture_image() == true)
				{
					this->aperture_vol = new Rpl_volume;

					this->aperture_vol->get_aperture()->set_center(this->get_aperture()->get_center());
					this->aperture_vol->get_aperture()->set_dim(this->get_aperture()->get_dim());
				
					this->aperture_vol->get_aperture()->set_distance(this->rpl_vol->get_aperture()->get_distance());
					this->aperture_vol->get_aperture()->set_spacing(this->rpl_vol->get_aperture()->get_spacing());

					this->aperture_vol->set_geometry (this->beam->get_source_position(), this->beam->get_isocenter_position(), this->get_aperture()->vup, this->get_aperture()->get_distance(), this->rpl_vol->get_aperture()->get_dim(), this->rpl_vol->get_aperture()->get_center(), this->get_aperture()->get_spacing(), this->get_step_length());

					this->aperture_vol->set_ct(this->rpl_vol->get_ct());
					this->aperture_vol->set_ct_limit(this->rpl_vol->get_ct_limit());
					this->aperture_vol->compute_ray_data();
					this->aperture_vol->set_front_clipping_plane(this->rpl_vol->get_front_clipping_plane());
					this->aperture_vol->set_back_clipping_plane(this->rpl_vol->get_back_clipping_plane());
					this->aperture_vol->compute_rpl_void();

					this->aperture_vol->compute_volume_aperture(this->get_aperture());
				}

                compute_sigmas(this, ppp->E0, sigma_max, "large", margins);				

                build_hong_grid(&area, &xy_grid, radius_sample, theta_sample);
                compute_dose_ray_shackleford(dose_vol, this, ppp, &area, &xy_grid, radius_sample, theta_sample);
            }
            printf("dose computed\n");
        }
    }
    if (this->beam->get_flavor() == 'a') // pull algorithm
    {     
        /* if (this->get_debug()) {
           rpl_vol->save ("beam_debug/depth_vol.mha");
           beam->dump ("beam_debug");
           }*/

		/* Create 3D aperture volume */
		if (this->get_aperture()->have_aperture_image() == true)
		{
			this->aperture_vol = new Rpl_volume;
			this->aperture_vol->get_aperture()->set_center(this->get_aperture()->get_center());
			this->aperture_vol->get_aperture()->set_dim(this->get_aperture()->get_dim());
			
			this->aperture_vol->get_aperture()->set_distance(this->rpl_vol->get_aperture()->get_distance());
			this->aperture_vol->get_aperture()->set_spacing(this->rpl_vol->get_aperture()->get_spacing());
			
			this->aperture_vol->set_geometry (this->beam->get_source_position(), this->beam->get_isocenter_position(), this->get_aperture()->vup, this->get_aperture()->get_distance(), this->rpl_vol->get_aperture()->get_dim(), this->rpl_vol->get_aperture()->get_center(), this->get_aperture()->get_spacing(), this->get_step_length());

			this->aperture_vol->set_ct(this->rpl_vol->get_ct());
			this->aperture_vol->set_ct_limit(this->rpl_vol->get_ct_limit());
			this->aperture_vol->compute_ray_data();
			this->aperture_vol->set_front_clipping_plane(this->rpl_vol->get_front_clipping_plane());
			this->aperture_vol->set_back_clipping_plane(this->rpl_vol->get_back_clipping_plane());
			this->aperture_vol->compute_rpl_void();

			this->aperture_vol->compute_volume_aperture(this->get_aperture());
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

					if (this->get_aperture()->have_aperture_image() == true && this->aperture_vol->get_rgdepth(ct_xyz) < .999)
					{
						continue;
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
                    case 'e':
                        dose = dose_hong_maxime (ct_xyz, ct_ijk, this);
                        break;
                    }

                    /* Insert the dose into the dose volume */
                    idx = volume_index (dose_vol->dim, ct_ijk);
                    dose_img[idx] = dose;
                }
            }
        }
        display_progress ((float)idx, (float)ct_vol->npix);
    }

    Plm_image::Pointer dose = Plm_image::New();
    dose->set_volume (dose_vol);
    d_ptr->dose = dose;

    printf ("Sigma conversion: %f seconds\n", time_sigma_conv);
    printf ("Dose calculation: %f seconds\n", time_dose_calc);
    printf ("Dose reformat: %f seconds\n", time_dose_reformat);
    printf ("Dose overhead: %f seconds\n", time_dose_misc);
}

void 
Ion_plan::dose_volume_create(Volume* dose_volume, float* sigma_max, Rpl_volume* volume, double range)
{
    /* we want to add extra margins around our volume take into account the dose that will be scattered outside of the rpl_volume */
    /* A 3 sigma margin is applied to the front_back volume, and the size of our volume will be the projection of this shape on the back_clipping_plane */
    
    float ap_ul_pixel[3]; // coordinates in the BEV (rpl_volume) volume
    float proj_pixel[3]; // coordinates of the ap_ul_pixel + 3 sigma margins on the back clipping plane
    float first_pixel[3]; // coordinates of the first_pixel of the volume to be created
	plm_long dim[3] = {0,0,0};
	float offset[3] = {0,0,0};
	float spacing[3] = {0,0,0};
	plm_long npix = 0;
	const float dc[9] = {
		dose_volume->get_direction_cosines()[0], dose_volume->get_direction_cosines()[1], dose_volume->get_direction_cosines()[2], 
		dose_volume->get_direction_cosines()[3], dose_volume->get_direction_cosines()[4], dose_volume->get_direction_cosines()[5], 
		dose_volume->get_direction_cosines()[6], dose_volume->get_direction_cosines()[7], dose_volume->get_direction_cosines()[8]};

    float sigma_margins = 3 * *sigma_max;
    double back_clip_useful = volume->compute_farthest_penetrating_ray_on_nrm(range) +10; // after this the volume will be void, the particules will not go farther + 2mm of margins

    ap_ul_pixel[0] = -volume->get_aperture()->get_center()[0]*volume->get_aperture()->get_spacing()[0];
    ap_ul_pixel[1] = -volume->get_aperture()->get_center()[1]*volume->get_aperture()->get_spacing()[1];
    ap_ul_pixel[2] = volume->get_aperture()->get_distance();

    proj_pixel[0] = (ap_ul_pixel[0] - sigma_margins)*(back_clip_useful + volume->get_aperture()->get_distance()) / volume->get_aperture()->get_distance();
    proj_pixel[1] = (ap_ul_pixel[1] - sigma_margins)*(back_clip_useful + volume->get_aperture()->get_distance()) / volume->get_aperture()->get_distance();
    proj_pixel[2] = back_clip_useful + volume->get_aperture()->get_distance();

    /* We build a matrix that starts from the proj_pixel projection on the front_clipping_plane */
    first_pixel[0] = floor(proj_pixel[0]);
    first_pixel[1] = floor(proj_pixel[1]);
    first_pixel[2] = floor(volume->get_front_clipping_plane() +volume->get_aperture()->get_distance());

    for (int i = 0; i < 3; i++)
    {
        offset[i] = first_pixel[i];
        if (i != 2)
        {   
            spacing[i] = 1;
            //spacing[i] = volume->get_aperture()->get_spacing(i); would be better...? pblm of lost lateral scattering for high resolution....
            dim[i] = (plm_long) (2*abs(first_pixel[i]/spacing[i])+1);
        }
        else
        {
            spacing[i] = volume->get_proj_volume()->get_step_length();
            dim[i] = (plm_long) ((back_clip_useful - volume->get_front_clipping_plane())/spacing[i] + 1);
        }
    }

    npix = dim[0]*dim[1]*dim[2];

	dose_volume->create(dim, offset, spacing, dc, PT_FLOAT,1);
}

