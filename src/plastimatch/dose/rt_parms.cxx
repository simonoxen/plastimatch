/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plm_config.h"
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "aperture.h"
#include "parameter_parser.h"
#include "plm_image.h"
#include "plm_math.h"
#include "print_and_exit.h"
#include "rpl_volume.h"
#include "rt_beam.h"
#include "rt_depth_dose.h"
#include "rt_parms.h"
#include "rt_plan.h"
#include "rt_sobp.h"
#include "string_util.h"

class Rt_parms_private {
public:

    /* [PEAK] */
    double E0;
    double spread;
    double weight;
    double max_depth;
    double depth_res;
    std::string bragg_curve;
    
    /* Other parameters not directly defined by config the config file but necessary for the beam creation */
    Rt_plan* rt_plan;
    int beam_number; /* contains the number of the beam in the vector<Rt_beam*> beam_storage */
    Rt_sobp::Pointer sobp;
    bool have_prescription;
    bool ap_have_origin;
    bool have_manual_peaks;

public:
    Rt_parms_private ()
    {
        /* [PEAK] */
        this->E0 = 100.;
        this->spread = 1.;
        this->weight = 1.;
        this->max_depth = 800.0f;
        this->depth_res = 1.f;
        this->bragg_curve ="";

        /* Other parameters not directly defined by config the config 
           file but necessary for the beam creation */
        this->rt_plan = 0;
        this->beam_number = -1;
        this->sobp = Rt_sobp::New();
        this->have_prescription = false;
        this->ap_have_origin = false;
        this->have_manual_peaks = false;
    }
};

Rt_parms::Rt_parms ()
{
    d_ptr = new Rt_parms_private;
}

Rt_parms::Rt_parms (Rt_plan* rt_plan)
{
    d_ptr = new Rt_parms_private;
    d_ptr->rt_plan = rt_plan;
}

Rt_parms::~Rt_parms ()
{
    delete d_ptr;
}

static void
print_usage (void)
{
    printf (
        "Usage: proton_dose [options] config_file\n"
        "Options:\n"
        " --debug           Create various debug files\n"
    );
    exit (1);
}

class Rt_parms_parser : public Parameter_parser
{
public:
    Rt_parms *rp;
public:
    Rt_parms_parser (Rt_parms *rp)
    {
        this->rp = rp;
    }
public:
    virtual Plm_return_code begin_section (
        const std::string& section)
    {
        if (section == "GLOBAL") {
            return PLM_SUCCESS;
        }
        if (section == "COMMENT") {
            return PLM_SUCCESS;
        }
        if (section == "PLAN") {
            return PLM_SUCCESS;
        }
        if (section == "BEAM") {
            rp->append_beam ();
            return PLM_SUCCESS;
        }
        if (section == "PEAK") {
            return PLM_SUCCESS;
        }

        /* else, unknown section */
        return PLM_ERROR;
    }
    virtual Plm_return_code end_section (
        const std::string& section)
    {
        if (section == "PEAK") {
            rp->append_peak ();
            return PLM_SUCCESS;
        }
        return PLM_SUCCESS;
    }
    virtual Plm_return_code set_key_value (
        const std::string& section,
        const std::string& key,
        const std::string& val)
    {
        return this->rp->set_key_value (section, key, val);
    }
};

void 
Rt_parms::set_rt_plan (Rt_plan *rt_plan)
{
    d_ptr->rt_plan = rt_plan;
}

void 
Rt_parms::append_beam ()
{
    d_ptr->rt_plan->append_beam ();
}

void 
Rt_parms::append_peak ()
{
    Rt_beam *rt_beam = d_ptr->rt_plan->get_last_rt_beam ();
    if (!rt_beam) {
        return;
    }
	rt_beam->set_have_manual_peaks(true);
    rt_beam->add_peak (
        d_ptr->E0, d_ptr->spread, d_ptr->depth_res, 
        d_ptr->max_depth, d_ptr->weight);
}

#if defined (commentout)
void 
save_beam_parameters(int i, int section)
{
    /* SETTINGS */
    if (section == 2)
    {
        d_ptr->rt_plan->beam_storage[i]->set_aperture_out(d_ptr->output_aperture_fn);			
        d_ptr->rt_plan->beam_storage[i]->set_proj_dose_out(d_ptr->output_proj_dose_fn);
        d_ptr->rt_plan->beam_storage[i]->set_proj_img_out(d_ptr->output_proj_img_fn);
        d_ptr->rt_plan->beam_storage[i]->set_range_compensator_out(d_ptr->output_range_compensator_fn);
        d_ptr->rt_plan->beam_storage[i]->set_sigma_out(d_ptr->output_sigma_fn.c_str());
        d_ptr->rt_plan->beam_storage[i]->set_wed_out(d_ptr->output_wed_fn.c_str());
        d_ptr->rt_plan->beam_storage[i]->set_particle_type(d_ptr->part);
        d_ptr->rt_plan->beam_storage[i]->set_detail(d_ptr->detail);
        d_ptr->rt_plan->beam_storage[i]->set_beamWeight(d_ptr->beam_weight);
        d_ptr->rt_plan->beam_storage[i]->set_source_position(d_ptr->src);
        d_ptr->rt_plan->beam_storage[i]->set_isocenter_position(d_ptr->isocenter);
        d_ptr->rt_plan->beam_storage[i]->set_sobp_prescription_min_max(d_ptr->prescription_min, d_ptr->prescription_max);
        d_ptr->rt_plan->beam_storage[i]->get_aperture()->set_distance(d_ptr->ap_offset);
        if (d_ptr->ap_have_origin) 
        {
            d_ptr->rt_plan->beam_storage[i]->get_aperture()->set_origin(d_ptr->ap_origin);
        }
        d_ptr->rt_plan->beam_storage[i]->get_aperture()->set_dim(d_ptr->ires);
        d_ptr->rt_plan->beam_storage[i]->get_aperture()->set_spacing(d_ptr->ap_spacing);
        d_ptr->rt_plan->beam_storage[i]->set_source_size(d_ptr->source_size);

#if defined (commentout)
        /* GCS TODO: This logic belongs elsewhere */
        if (d_ptr->target_fn == "") 
        {
            if (d_ptr->ap_filename != "") 
            {
                d_ptr->rt_plan->beam_storage[i]->set_aperture_in(d_ptr->ap_filename.c_str());
            }
            if (d_ptr->rc_filename != "") 
            {
                d_ptr->rt_plan->beam_storage[i]->set_range_compensator_in (d_ptr->rc_filename.c_str());
            }
        }
#endif
        d_ptr->rt_plan->beam_storage[i]->set_smearing(d_ptr->smearing);
        d_ptr->rt_plan->beam_storage[i]->set_proximal_margin(d_ptr->proximal_margin);
        d_ptr->rt_plan->beam_storage[i]->set_distal_margin(d_ptr->distal_margin);
        d_ptr->rt_plan->beam_storage[i]->set_have_prescription(d_ptr->have_prescription);
        d_ptr->rt_plan->beam_storage[i]->set_photon_energy(d_ptr->photon_energy);
    }

    /* PEAKS */
    else if (section == 3)
    {
        if (d_ptr->bragg_curve =="")
        {
            d_ptr->rt_plan->beam_storage[i]->set_have_manual_peaks(
                d_ptr->have_manual_peaks);
            d_ptr->rt_plan->beam_storage[i]->add_peak(
                d_ptr->E0, d_ptr->spread, d_ptr->depth_res, 
                d_ptr->max_depth, d_ptr->weight);
        }
        else
        {
            printf("ERROR: bragg curve already defined by bragg_curve file - impossible to optimize a SOBP from peaks");
        }
    }
}
#endif

Plm_return_code
Rt_parms::set_key_value (
    const std::string& section,
    const std::string& key, 
    const std::string& val)
{
    if (section == "COMMENT" || section == "GLOBAL") {
        return PLM_SUCCESS;
    }

    /* **** PLAN **** */
    if (section == "PLAN") {
        if (key == "patient") {
            d_ptr->rt_plan->set_patient (val);
        }
        else if (key == "target") {
            d_ptr->rt_plan->set_target (val);
         }
        else if (key == "threading") {
            Threading threading = THREADING_CPU_OPENMP;
            if (val == "single") {
                threading = THREADING_CPU_SINGLE;
            }
            else if (val == "openmp") {
#if (OPENMP_FOUND)
                threading = THREADING_CPU_OPENMP;
#else
                threading = THREADING_CPU_SINGLE;
#endif
            }
            else if (val == "cuda") {
#if (CUDA_FOUND)
                threading = THREADING_CUDA;
#elif (OPENMP_FOUND)
                threading = THREADING_CPU_OPENMP;
#else
                threading = THREADING_CPU_SINGLE;
#endif
            }
            else {
                goto error_exit;
            }
            d_ptr->rt_plan->set_threading (threading);
        }
        else if (key == "dose_out") {
            d_ptr->rt_plan->set_output_dose (val);
        }
        else if (key == "debug") {
            d_ptr->rt_plan->set_debug (string_value_true (val));
        }
        else if (key == "dose_prescription") {
            float norm_dose;
            if (sscanf (val.c_str(), "%f", &norm_dose) != 1) {
                goto error_exit;
            }
            d_ptr->rt_plan->set_normalization_dose (norm_dose);
        }
        else {
            goto error_exit;
        }
        return PLM_SUCCESS;
    }

    /* **** BEAM **** */
    if (section == "BEAM") {
        Rt_beam *rt_beam = d_ptr->rt_plan->get_last_rt_beam ();

        if (key == "flavor") {
            if (val.length() >= 1) {
                rt_beam->set_flavor (val[0]);
            } else {
                goto error_exit;
            } 
        }
        else if (key == "homo_approx") {
            if (val.length() >= 1) {
                rt_beam->set_homo_approx (val[0]);
            } else {
                goto error_exit;
            } 
        }
        else if (key == "ray_step") {
            float step_length;
            if (sscanf (val.c_str(), "%f", &step_length) != 1) {
                goto error_exit;
            }
            rt_beam->set_step_length (step_length);
        }


        else if (key == "aperture_out") {
            rt_beam->set_aperture_out (val);
        }
        else if (key == "proj_dose_out") {
            rt_beam->set_proj_dose_out (val);
        }
        else if (key == "proj_img_out") {
            rt_beam->set_proj_img_out (val);
        }
        else if (key == "rc_out") {
            rt_beam->set_range_compensator_out (val);
        }
        else if (key == "sigma_out") {
            rt_beam->set_sigma_out (val);
        }
        else if (key == "wed_out") {
            rt_beam->set_wed_out (val);
        }
        else if (key == "beam_type") {
            Particle_type part = particle_type_parse (val);
            if (part == PARTICLE_TYPE_UNKNOWN) {
                goto error_exit;
            }
            rt_beam->set_particle_type (part);
        }
        else if (key == "detail") {
            if (val == "low") {
                rt_beam->set_detail (1);
            }
            else if (val == "high") {
                rt_beam->set_detail (0);
            }
            else {
                goto error_exit;
            }
        }
        else if (key == "beam_weight") {
            float beam_weight;
            if (sscanf (val.c_str(), "%f", &beam_weight) != 1) {
                goto error_exit;
            }
            rt_beam->set_beam_weight (beam_weight);
        }
        else if (key == "depth_dose_z_max") {
            if (sscanf (val.c_str(), "%lf", &(d_ptr->max_depth)) != 1) {
                goto error_exit;
            }
        }
        else if (key == "depth_dose_z_res") {
            if (sscanf (val.c_str(), "%lf", &(d_ptr->depth_res)) != 1) {
                goto error_exit;
            }
        }
        else if (key == "source") {
            float src[3];
            int rc = sscanf (val.c_str(), "%f %f %f", 
                &src[0], &src[1], &src[2]);
            if (rc != 3) {
                goto error_exit;
            }
            rt_beam->set_source_position (src);
        }
        else if (key == "isocenter") {
            float isocenter[3];
            int rc = sscanf (val.c_str(), "%f %f %f", &isocenter[0],
                &isocenter[1], &isocenter[2]);
            if (rc != 3) {
                goto error_exit;
            }
            rt_beam->set_isocenter_position (isocenter);
        }
        else if (key == "prescription_min") {
            float prescription_min;
            int rc = sscanf (val.c_str(), "%f", &prescription_min);
            if (rc != 1) {
                goto error_exit;
            }
            rt_beam->set_prescription_min (prescription_min);
        }
        else if (key == "prescription_max") {
            float prescription_max;
            int rc = sscanf (val.c_str(), "%f", &prescription_max);
            if (rc != 1) {
                goto error_exit;
            }
            rt_beam->set_prescription_max (prescription_max);
        }
        else if (key == "aperture_up") {
            float vup[3];
            int rc = sscanf (val.c_str(), "%f %f %f", 
                &vup[0], &vup[1], &vup[2]);
            if (rc != 3) {
                goto error_exit;
            }
            rt_beam->set_aperture_vup (vup);
        }
        else if (key == "aperture_offset") {
            float ap_distance;
            if (sscanf (val.c_str(), "%f", &ap_distance) != 1) {
                goto error_exit;
            }
            rt_beam->set_aperture_distance (ap_distance);
        }
        else if (key == "aperture_origin") {
            float ap_origin[2];
            int rc = sscanf (val.c_str(), "%f %f", 
                &ap_origin[0], &ap_origin[1]);
            if (rc != 2) {
                goto error_exit;
            }
            rt_beam->set_aperture_origin (ap_origin);
        }
        else if (key == "aperture_resolution") {
            int ap_dim[2];
            int rc = sscanf (val.c_str(), "%i %i", &ap_dim[0], &ap_dim[1]);
            if (rc != 2) {
                goto error_exit;
            }
            rt_beam->set_aperture_resolution (ap_dim);
        }
        else if (key == "aperture_spacing") {
            float ap_spacing[2];
            int rc = sscanf (val.c_str(), "%f %f", 
                &ap_spacing[0], &ap_spacing[1]);
            if (rc != 2) {
                goto error_exit;
            }
            rt_beam->set_aperture_spacing (ap_spacing);
        }
        else if (key == "source_size") {
            float source_size;
            if (sscanf (val.c_str(), "%f", &source_size) != 1) {
                goto error_exit;
            }
            rt_beam->set_source_size (source_size);
        }
        else if (key == "aperture_file_in") {
            rt_beam->set_aperture_in (val);
        }
        else if (key == "range_compensator_file_in") {
            rt_beam->set_range_compensator_in (val);
        }
        else if (key == "aperture_smearing") {
            float smearing;
            if (sscanf (val.c_str(), "%f", &smearing) != 1) {
                goto error_exit;
            }
            rt_beam->set_smearing (smearing);
        }
        else if (key == "proximal_margin") {
            float proximal_margin;
            if (sscanf (val.c_str(), "%f", &proximal_margin) != 1) {
                goto error_exit;
            }
            rt_beam->set_proximal_margin (proximal_margin);
        }
        else if (key == "distal_margin") {
            float distal_margin;
            if (sscanf (val.c_str(), "%f", &distal_margin) != 1) {
                goto error_exit;
            }
            rt_beam->set_distal_margin (distal_margin);
        }
        else if (key == "energy_x") {
            float photon_energy;
            if (sscanf (val.c_str(), "%f", &photon_energy) != 1) {
                goto error_exit;
            }
            rt_beam->set_photon_energy (photon_energy);
        }
        else {
            goto error_exit;
        }
        return PLM_SUCCESS;
    }

    if (section == "PEAK") {
        if (key == "energy") {
            if (sscanf (val.c_str(), "%lf", &(d_ptr->E0)) != 1) {
                goto error_exit;
            }
        }
        else if (key == "spread") {
            if (sscanf (val.c_str(), "%lf", &(d_ptr->spread)) != 1) {
                goto error_exit;
            }
        }
        else if (key == "weight") {
            if (sscanf (val.c_str(), "%lf", &(d_ptr->weight)) != 1) {
                goto error_exit;
            }
        }
        else if (key == "bragg_curve") {
#if defined (commentout_TODO)
            d_ptr->rt_plan->beam->load (val);
#endif
        }
        else {
            goto error_exit;
        }
        return PLM_SUCCESS;
    }

    print_and_exit ("Unknown section value: %s\n", section.c_str());
    return PLM_ERROR;

error_exit:
    print_and_exit ("Unknown (key,val) combination: (%s,%s)\n", 
        key.c_str(), val.c_str());
    return PLM_ERROR;
}

#if defined (commentout_TODO)
void
Rt_parms::handle_end_of_section (int section)
{
    switch (section) {
    case 0:
        /* Reading before [PLAN] */
        break;
    case 1:
        /* [PLAN] */
        d_ptr->rt_plan->set_debug(d_ptr->debug);
        /* The other parameters are set at the beginning of the dose calculation */
        break;

    case 2:
        /* [BEAM] */
        Rt_beam* new_beam;
        new_beam = new Rt_beam;

        d_ptr->beam_number++; /* initialized to -1, at first run on the beam, beam_number = 0 */
        d_ptr->rt_plan->beam_storage.push_back(new_beam);

        d_ptr->sobp = Rt_sobp::New();
        /* We save the beam data, only the changes will be updated in the other sections */
        this->save_beam_parameters(d_ptr->beam_number, 2);

        d_ptr->have_manual_peaks = false;
        d_ptr->ap_have_origin = false;
        d_ptr->have_prescription = false;

        d_ptr->output_aperture_fn= "";
        d_ptr->output_proj_dose_fn = "";
        d_ptr->output_proj_img_fn = "";
        d_ptr->output_range_compensator_fn = "";
        d_ptr->output_sigma_fn = "";
        d_ptr->output_wed_fn = "";
        d_ptr->ap_filename= "";
        d_ptr->rc_filename = "";
        break;
    case 3:
        /* [PEAK] */
        d_ptr->have_manual_peaks = true;
        this->save_beam_parameters(d_ptr->beam_number, section);
        break;
    }
}
#endif

Plm_return_code
Rt_parms::parse_args (int argc, char** argv)
{
    int i;
    for (i=1; i<argc; i++) {
        if (argv[i][0] != '-') break;

        if (!strcmp (argv[i], "--debug")) {
            d_ptr->rt_plan->set_debug (true);
        }
        else {
            print_usage ();
            break;
        }
    }

    if (!argv[i]) {
        print_usage ();
    }

    Rt_parms_parser rpp (this);
    return rpp.parse_config_file (argv[i]);
}
