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

    /* [Plan] */
	std::string input_ct_fn;    /* input:  patient volume */
	std::string target_fn;
    Threading threading;
	std::string output_dose_fn;
    int debug;            /* 1 = debug mode */
	float dose_norm;      /* normalization dose intensity */

	/* [BEAM] */

	/* [BEAM SETTINGS] */
	char flavor;
	char homo_approx;
	float ray_step;
	std::string output_aperture_fn;
	std::string output_proj_dose_fn;
    std::string output_proj_img_fn;
    std::string output_range_compensator_fn;
    std::string output_sigma_fn;
    std::string output_wed_fn;
	Particle_type part;
    int detail;           /* 0 = full detail */ /* 1 = only consider voxels in beam path */
	float beam_weight;

	/* [BEAM GEOMETRY] */
	float src[3];
    float isocenter[3];
	float prescription_min;
    float prescription_max;
	
    /* [BEAM APERTURE] */
	float vup[3];
	float ap_offset;
    float ap_origin[2];
	int ires[2];
	float ap_spacing[2];
	float source_size;
	std::string ap_filename;
    std::string rc_filename;
    float smearing;
    float proximal_margin;
    float distal_margin;

	/* [PEAK] */
    double E0;
    double spread;
    double weight;
	double max_depth;
	double depth_res;
	std::string bragg_curve;
    
	/* [PHOTON ENERGY] */
	double photon_energy;

	/* Other parameters not directly defined by config the config file but necessary for the beam creation */
	Rt_plan::Pointer plan;
	int beam_number; /* contains the number of the beam in the vector<Rt_beam*> beam_storage */
	Rt_sobp::Pointer sobp;
	bool have_prescription;
	bool ap_have_origin;
	bool have_manual_peaks;

public:
    Rt_parms_private () {

        /* GCS FIX: Copy-paste with wed_parms.cxx */

		/* Plan */			
		std::string patient_fn = "";
		std::string target_fn = "";
		this->threading = THREADING_CPU_OPENMP;
		this->output_dose_fn ="";
		this->debug = 0;
		this->dose_norm = 1.0f;

		/* [BEAM SETTINGS] */
		this->flavor = 'a';
		this->homo_approx = 'n';
		this->ray_step = 1.0f;
		this->output_aperture_fn = "";
		this->output_proj_dose_fn = "";
		this->output_proj_img_fn = "";
		this->output_range_compensator_fn = "";
		this->output_sigma_fn = "";
		this->output_wed_fn = "";
		this->part = PARTICLE_TYPE_P;
		this->detail = 0;
		this->beam_weight = 1.f;

		/* [BEAM GEOMETRY] */
        this->src[0] = -1000.f;
        this->src[1] = 0.f;
        this->src[2] = 0.f;
        this->isocenter[0] = 0.f;
        this->isocenter[1] = 0.f;
        this->isocenter[2] = 0.f;
        this->prescription_min = 50.0f;
        this->prescription_max = 100.0f;

		/* [BEAM APERTURE] */
		this->vup[0] = 0.f;
        this->vup[1] = 0.f;
        this->vup[2] = 1.f;
		this->ap_offset = 100;
        this->ap_origin[0] = 0.;
        this->ap_origin[1] = 0.;
		this->ires[0] = 200;
        this->ires[1] = 200;
		this->ap_spacing[0] = 1.;
        this->ap_spacing[1] = 1.;
		this->source_size = 0.;
		this->ap_filename = "";
		this->rc_filename = "";
		this->smearing = 0.;
		this->proximal_margin = 0.;
        this->distal_margin = 0.;
        
        /* [PEAK] */
        this->E0 = 100.;
        this->spread = 1.;
		this->weight = 1.;
		this->max_depth = 800.0f;
        this->depth_res = 1.f;
		this->bragg_curve ="";

		/* [PHOTON ENERGY] */
		this->photon_energy = 6.;

		/* Other parameters not directly defined by config the config file but necessary for the beam creation */
		this->plan = Rt_plan::New ();
		this->beam_number = -1;
		this->sobp = Rt_sobp::New();
		this->have_prescription = false;
		this->ap_have_origin = false;
		this->have_manual_peaks = false;
    }
};

Rt_parms::Rt_parms ()
{
    this->d_ptr = new Rt_parms_private;
}

Rt_parms::~Rt_parms ()
{
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

void 
Rt_parms::save_beam_parameters(int i, int section)
{
	if (i >= d_ptr->plan->beam_storage.size() || i < 0)
	{
		printf("invalid beam_storage size - check your config file");
		return;
	}
	else
	{
		/* SETTINGS */
		if (section == 2)
		{
			d_ptr->plan->beam_storage[i]->set_flavor(d_ptr->flavor);
			d_ptr->plan->beam_storage[i]->set_homo_approx(d_ptr->homo_approx);
			d_ptr->plan->beam_storage[i]->set_step_length(d_ptr->ray_step);
			d_ptr->plan->beam_storage[i]->set_aperture_out(d_ptr->output_aperture_fn);			
			d_ptr->plan->beam_storage[i]->set_proj_dose_out(d_ptr->output_proj_dose_fn);
			d_ptr->plan->beam_storage[i]->set_proj_img_out(d_ptr->output_proj_img_fn);
			d_ptr->plan->beam_storage[i]->set_range_compensator_out(d_ptr->output_range_compensator_fn);
			d_ptr->plan->beam_storage[i]->set_sigma_out(d_ptr->output_sigma_fn.c_str());
			d_ptr->plan->beam_storage[i]->set_wed_out(d_ptr->output_wed_fn.c_str());
			d_ptr->plan->beam_storage[i]->set_particle_type(d_ptr->part);
			d_ptr->plan->beam_storage[i]->set_detail(d_ptr->detail);
			d_ptr->plan->beam_storage[i]->set_beamWeight(d_ptr->beam_weight);
			d_ptr->plan->beam_storage[i]->set_source_position(d_ptr->src);
			d_ptr->plan->beam_storage[i]->set_isocenter_position(d_ptr->isocenter);
			d_ptr->plan->beam_storage[i]->set_sobp_prescription_min_max(d_ptr->prescription_min, d_ptr->prescription_max);
			d_ptr->plan->beam_storage[i]->get_aperture()->vup[0] = d_ptr->vup[0];
			d_ptr->plan->beam_storage[i]->get_aperture()->vup[1] = d_ptr->vup[1];
			d_ptr->plan->beam_storage[i]->get_aperture()->vup[2] = d_ptr->vup[2];
			d_ptr->plan->beam_storage[i]->get_aperture()->set_distance(d_ptr->ap_offset);
			if (d_ptr->ap_have_origin) 
			{
				d_ptr->plan->beam_storage[i]->get_aperture()->set_origin(d_ptr->ap_origin);
			}
			d_ptr->plan->beam_storage[i]->get_aperture()->set_dim(d_ptr->ires);
			d_ptr->plan->beam_storage[i]->get_aperture()->set_spacing(d_ptr->ap_spacing);
			d_ptr->plan->beam_storage[i]->set_source_size(d_ptr->source_size);
			if (d_ptr->target_fn == "") 
			{
				if (d_ptr->ap_filename != "") 
				{
					d_ptr->plan->beam_storage[i]->set_aperture_in(d_ptr->ap_filename.c_str());
				}
				if (d_ptr->rc_filename != "") 
				{
					d_ptr->plan->beam_storage[i]->set_range_compensator_in (d_ptr->rc_filename.c_str());
				}
			}			
			d_ptr->plan->beam_storage[i]->set_smearing(d_ptr->smearing);
			d_ptr->plan->beam_storage[i]->set_proximal_margin(d_ptr->proximal_margin);
			d_ptr->plan->beam_storage[i]->set_distal_margin(d_ptr->distal_margin);
			d_ptr->plan->beam_storage[i]->set_have_prescription(d_ptr->have_prescription);
			d_ptr->plan->beam_storage[i]->set_photon_energy(d_ptr->photon_energy);
		}

		/* PEAKS */
		else if (section == 3)
		{
			if (d_ptr->bragg_curve =="")
			{
				d_ptr->plan->beam_storage[i]->set_have_manual_peaks(d_ptr->have_manual_peaks);
				d_ptr->plan->beam_storage[i]->add_peak(d_ptr->E0, d_ptr->spread, d_ptr->depth_res, d_ptr->max_depth, d_ptr->weight);
			}
			else
			{
				printf("ERROR: bragg curve already defined by bragg_curve file - impossible to optimize a SOBP from peaks");
			}
		}
	}
}

void
Rt_parms::print_verif(Rt_plan::Pointer plan)
{
	printf("\n [PLAN]");
	printf("\n patient : %s", d_ptr->input_ct_fn.c_str());
	printf("\n target : %s", d_ptr->target_fn.c_str());
	printf("\n dose_out : %s", d_ptr->output_dose_fn.c_str());
	printf("\n debug : %d", plan->get_debug());
	printf("\n dose norm : %lg", plan->get_normalization_dose());

	printf("\n \n [SETTINGS]");
	printf("\n flavor: "); for (int i = 0; i <= d_ptr->beam_number; i++) {printf("%c ** ", plan->beam_storage[i]->get_flavor());}
	printf("\n homo_approx: "); for (int i = 0; i <= d_ptr->beam_number; i++) {printf("%c ** ", plan->beam_storage[i]->get_homo_approx());}
	printf("\n ray_step: "); for (int i = 0; i <= d_ptr->beam_number; i++) {printf("%lg ** ", plan->beam_storage[i]->get_step_length());}
	printf("\n aperture_out: "); for (int i = 0; i <= d_ptr->beam_number; i++) {printf("%s ** ", plan->beam_storage[i]->get_aperture_out().c_str());}
	printf("\n proj_dose_out: "); for (int i = 0; i <= d_ptr->beam_number; i++) {printf("%s ** ", plan->beam_storage[i]->get_proj_dose_out().c_str());}
	printf("\n proj_img_out: "); for (int i = 0; i <= d_ptr->beam_number; i++) {printf("%s ** ", plan->beam_storage[i]->get_proj_img_out().c_str());}
	printf("\n range_comp_out: "); for (int i = 0; i <= d_ptr->beam_number; i++) {printf("%s ** ", plan->beam_storage[i]->get_range_compensator_out().c_str());}
	printf("\n sigma_out: "); for (int i = 0; i <= d_ptr->beam_number; i++) {printf("%s ** ", plan->beam_storage[i]->get_sigma_out().c_str());}
	printf("\n wed_out: "); for (int i = 0; i <= d_ptr->beam_number; i++) {printf("%s ** ", plan->beam_storage[i]->get_wed_out().c_str());}
	printf("\n part_type: "); for (int i = 0; i <= d_ptr->beam_number; i++) {printf("%d ** ", plan->beam_storage[i]->get_particle_type());}
	printf("\n detail: "); for (int i = 0; i <= d_ptr->beam_number; i++) {printf("%d ** ", plan->beam_storage[i]->get_detail());}
	printf("\n beam_weight: "); for (int i = 0; i <= d_ptr->beam_number; i++) {printf("%d ** ", plan->beam_storage[i]->get_beamWeight());}
	//printf("\n max_depth: "); for (int i = 0; i <= d_ptr->beam_number; i++) { printf("P%d %d",i, plan->beam_storage[i]->get_sobp()->get_num_peaks()); for (int j = 0; j < plan->beam_storage[i]->get_sobp()->get_num_peaks(); j++) { printf(" %lg ** ", plan->beam_storage[i]->get_sobp()->get_depth_dose()[j]->dmax);}}
	//printf("\n depth_res: "); for (int i = 0; i <= d_ptr->beam_number; i++) { printf("P%d ",i); for (int j = 0; j < plan->beam_storage[i]->get_sobp()->get_num_peaks(); j++) { printf("%lg ** ", plan->beam_storage[i]->get_sobp()->get_depth_dose()[j]->dres);}}

	printf("\n \n [GEOMETRY & APERTURE]");
	printf("\n source: "); for (int i = 0; i <= d_ptr->beam_number; i++) {printf("%lg %lg %lg ** ", plan->beam_storage[i]->get_source_position()[0], plan->beam_storage[i]->get_source_position()[1], plan->beam_storage[i]->get_source_position()[2]);}
	printf("\n isocenter: "); for (int i = 0; i <= d_ptr->beam_number; i++) {printf("%lg %lg %lg ** ", plan->beam_storage[i]->get_isocenter_position()[0], plan->beam_storage[i]->get_isocenter_position()[1], plan->beam_storage[i]->get_isocenter_position()[2]);}
	printf("\n vup: "); for (int i = 0; i <= d_ptr->beam_number; i++) {printf("%lg %lg %lg ** ", plan->beam_storage[i]->get_aperture()->vup[0], plan->beam_storage[i]->get_aperture()->vup[1], plan->beam_storage[i]->get_aperture()->vup[2]);}
	printf("\n offset: "); for (int i = 0; i <= d_ptr->beam_number; i++) {printf("%lg ** ", plan->beam_storage[i]->get_aperture()->get_distance());}
	printf("\n ap_origin: "); for (int i = 0; i <= d_ptr->beam_number; i++) {printf("%lg %lg ** ", plan->beam_storage[i]->get_aperture()->get_center()[0], plan->beam_storage[i]->get_aperture()->get_center()[1]);}
	printf("\n i_res: "); for (int i = 0; i <= d_ptr->beam_number; i++) {printf("%d %d ** ", plan->beam_storage[i]->get_aperture()->get_dim()[0], plan->beam_storage[i]->get_aperture()->get_dim()[1]);}
	printf("\n spacing: "); for (int i = 0; i <= d_ptr->beam_number; i++) {printf("%lg %lg ** ", plan->beam_storage[i]->get_aperture()->get_spacing()[0], plan->beam_storage[i]->get_aperture()->get_spacing()[1]);}
	printf("\n source_size: "); for (int i = 0; i <= d_ptr->beam_number; i++) {printf("%lg ** ", plan->beam_storage[i]->get_source_size());}
	printf("\n ap_file_in: "); for (int i = 0; i <= d_ptr->beam_number; i++) {printf("%s ** ", plan->beam_storage[i]->get_aperture_in().c_str());}
	printf("\n rc_file_in: "); for (int i = 0; i <= d_ptr->beam_number; i++) {printf("%s ** ", plan->beam_storage[i]->get_range_compensator_in().c_str());}
	printf("\n smearing: "); for (int i = 0; i <= d_ptr->beam_number; i++) {printf("%lg ** ", plan->beam_storage[i]->get_smearing());}
	printf("\n prox_margin: "); for (int i = 0; i <= d_ptr->beam_number; i++) {printf("%lg ** ", plan->beam_storage[i]->get_proximal_margin());}
	printf("\n dist_margin: "); for (int i = 0; i <= d_ptr->beam_number; i++) {printf("%lg ** ", plan->beam_storage[i]->get_distal_margin());}

	printf("\n \n [PEAK]");
	//printf("\n E0: "); for (int i = 0; i <= d_ptr->beam_number; i++) { printf("P%d ",i); for (int j = 0; j < plan->beam_storage[i]->get_sobp()->get_num_peaks(); j++) { printf("%lg ** ", plan->beam_storage[i]->get_sobp()->get_depth_dose()[j]->E0);}}
	//printf("\n spread: "); for (int i = 0; i <= d_ptr->beam_number; i++) { printf("P%d ",i); for (int j = 0; j < plan->beam_storage[i]->get_sobp()->get_depth_dose().size(); j++) { printf("%lg ** ", plan->beam_storage[i]->get_sobp()->get_depth_dose()[j]->spread);}}
	//printf("\n weight: "); for (int i = 0; i <= d_ptr->beam_number; i++) { printf("P%d ",i); for (int j = 0; j < plan->beam_storage[i]->get_sobp()->get_depth_dose().size(); j++) { printf("%lg ** ", plan->beam_storage[i]->get_sobp()->get_depth_dose()[j]->weight);}}

	printf("\n \n [PHOTON_ENERGY]");
	printf("\n photon energy: "); for (int i = 0; i <= d_ptr->beam_number; i++) {printf("%lg ** ", plan->beam_storage[i]->get_photon_energy());}
}

int
Rt_parms::set_key_val (
    const char* key, 
    const char* val, 
    int section
)
{
    switch (section) {

	/* Reading before first flag [PLAN] */
	case 0:

	/* [PLAN] */
	case 1:
		if (!strcmp (key, "patient")) {
            d_ptr->input_ct_fn = val;
        }
		else if (!strcmp (key, "target")) {
            d_ptr->target_fn = val;
        }
		else if (!strcmp (key, "threading")) {
            if (!strcmp (val,"single")) {
                d_ptr->threading = THREADING_CPU_SINGLE;
            }
            else if (!strcmp (val,"openmp")) {
#if (OPENMP_FOUND)
                d_ptr->threading = THREADING_CPU_OPENMP;
#else
                this->threading = THREADING_CPU_SINGLE;
#endif
            }
            else if (!strcmp (val,"cuda")) {
#if (CUDA_FOUND)
                this->threading = THREADING_CUDA;
#elif (OPENMP_FOUND)
                d_ptr->threading = THREADING_CPU_OPENMP;
#else
                this->threading = THREADING_CPU_SINGLE;
#endif
            }
            else {
                goto error_exit;
            }
        }
		else if (!strcmp (key, "dose_out")) {
            d_ptr->output_dose_fn = val;
        }
		else if (!strcmp (key, "debug")) {
            if (sscanf (val, "%d", &d_ptr->debug) != 1) {
                goto error_exit;
            }
        }
		else if (!strcmp (key, "dose_prescription")) {
			if (sscanf (val, "%f", &d_ptr->dose_norm) != 1) {
                goto error_exit;
            }
        }
		else {
            goto error_exit;
        }
        break;

    /* [BEAM] */
    case 2:
        if (!strcmp (key, "flavor")) {
            if (strlen (val) >= 1) {
                d_ptr->flavor = val[0];
            } else {
                goto error_exit;
            } 
        }
        else if (!strcmp (key, "homo_approx")) {
            if (strlen (val) >= 1) {
              d_ptr->homo_approx = val[0];
            } else {
                goto error_exit;
            } 
        }
        else if (!strcmp (key, "ray_step")) {
            if (sscanf (val, "%f", &d_ptr->ray_step) != 1) {
                goto error_exit;
            }
        }
		else if (!strcmp (key, "aperture_out")) {
            d_ptr->output_aperture_fn = val;
        }
		else if (!strcmp (key, "proj_dose_out")) {
            d_ptr->output_proj_dose_fn = val;
        }
		else if (!strcmp (key, "proj_img_out")) {
            d_ptr->output_proj_img_fn = val;
		}
		else if (!strcmp (key, "rc_out")) {
            d_ptr->output_range_compensator_fn = val;
        }
		else if (!strcmp (key, "sigma_out")) {
            d_ptr->output_sigma_fn = val;
        }
		else if (!strcmp (key, "wed_out")) {
            d_ptr->output_wed_fn = val;
        }
		else if (!strcmp (key, "beam_type")) {
			if (!strcmp (val, "X")) {
				d_ptr->part = PARTICLE_TYPE_X;
			}
			else if (!strcmp (val, "P")) {
				d_ptr->part = PARTICLE_TYPE_P;
			}

			else if (!strcmp (val, "HE")) {
				d_ptr->part = PARTICLE_TYPE_HE;
			}
			else if (!strcmp (val, "LI")) {
				d_ptr->part = PARTICLE_TYPE_LI;
			}
			else if (!strcmp (val, "P")) {
				d_ptr->part = PARTICLE_TYPE_P;
			}
			else if (!strcmp (val, "BE")) {
				d_ptr->part = PARTICLE_TYPE_BE;
			}
			else if (!strcmp (val, "B")) {
				d_ptr->part = PARTICLE_TYPE_B;
			}
			else if (!strcmp (val, "C")) {
				d_ptr->part = PARTICLE_TYPE_C;
			}
			else if (!strcmp (val, "O")) {
				d_ptr->part = PARTICLE_TYPE_O;
			}
			else {
                goto error_exit;
            }
        }
        else if (!strcmp (key, "detail")) {
            if (!strcmp (val, "low")) {
                d_ptr->detail = 1;
            }
            else if (!strcmp (val, "high")) {
                d_ptr->detail = 0;
            }
            else {
                goto error_exit;
            }
        }
		else if (!strcmp (key, "beam_weight")) {
            if (sscanf (val, "%lf", &(d_ptr->beam_weight)) != 1) {
                goto error_exit;
            }
        }
		else if (!strcmp (key, "depth_dose_z_max")) {
            if (sscanf (val, "%lf", &(d_ptr->max_depth)) != 1) {
                goto error_exit;
            }
        }
        else if (!strcmp (key, "depth_dose_z_res")) {
            if (sscanf (val, "%lf", &(d_ptr->depth_res)) != 1) {
                goto error_exit;
            }
        }
		else if (!strcmp (key, "source")) {
            int rc = sscanf (val, "%f %f %f", 
                &d_ptr->src[0], &d_ptr->src[1], &d_ptr->src[2]);
            if (rc != 3) {
                goto error_exit;
            }
        }
		else if (!strcmp (key, "isocenter")) {
            int rc = sscanf (val, "%f %f %f", &d_ptr->isocenter[0],
                &d_ptr->isocenter[1], &d_ptr->isocenter[2]);
            if (rc != 3) {
                goto error_exit;
            }
		}
		else if (!strcmp (key, "prescription_min")) {
            int rc = sscanf (val, "%f", &d_ptr->prescription_min);
            if (rc != 1) {
                goto error_exit;
            }
            d_ptr->have_prescription = true;
        }
        else if (!strcmp (key, "prescription_max")) {
            int rc = sscanf (val, "%f", &d_ptr->prescription_max);
            if (rc != 1) {
                goto error_exit;
            }
            d_ptr->have_prescription = true;
        }
		else if (!strcmp (key, "aperture_up")) {
            if (sscanf (val, "%f %f %f", &d_ptr->vup[0], 
                    &d_ptr->vup[1], &d_ptr->vup[2]) != 3)
            {
                goto error_exit;
            }
        }
		else if (!strcmp (key, "aperture_offset")) {
            if (sscanf (val, "%f", &d_ptr->ap_offset) != 1) {
                goto error_exit;
            }
        }
		else if (!strcmp (key, "aperture_origin")) {
            if (sscanf (val, "%f %f", 
                    &d_ptr->ap_origin[0], &d_ptr->ap_origin[1]) != 2) {
                goto error_exit;
            }
            d_ptr->ap_have_origin = true;
        }
		else if (!strcmp (key, "aperture_resolution")) {
            if (sscanf (val, "%i %i", &d_ptr->ires[0], &d_ptr->ires[1]) != 2) {
                goto error_exit;
            }
        }
		else if (!strcmp (key, "aperture_spacing")) {
            if (sscanf (val, "%f %f", 
                    &d_ptr->ap_spacing[0], &d_ptr->ap_spacing[1]) != 2) {
                goto error_exit;
            }
        }
		else if (!strcmp (key, "source_size")) {
			if (sscanf (val, "%f", &d_ptr->source_size) !=1) {
				goto error_exit;
			}
		}
		else if (!strcmp (key, "aperture_file_in")) {
            d_ptr->ap_filename = val;
        }
        else if (!strcmp (key, "range_compensator_file_in")) {
            d_ptr->rc_filename = val;
        }
		else if (!strcmp (key, "aperture_smearing")) {
            if (sscanf (val, "%f", &d_ptr->smearing) != 1) {
                goto error_exit;
            }
        }
		else if (!strcmp (key, "proximal_margin")) {
            if (sscanf (val, "%f", &d_ptr->proximal_margin) != 1) {
                goto error_exit;
            }
        }
        else if (!strcmp (key, "distal_margin")) {
            if (sscanf (val, "%f", &d_ptr->distal_margin) != 1) {
                goto error_exit;
            }
        }
		else if (!strcmp (key, "energy_x")) {
			if (sscanf (val, "%f", &d_ptr->photon_energy) != 1) {
                goto error_exit;
            }
        }
        else {
            goto error_exit;
        }
        break;

		/* [PEAK] */
	case 3:
		if (!strcmp (key, "energy")) {
            if (sscanf (val, "%lf", &(d_ptr->E0)) != 1) {
                goto error_exit;
            }
        }
        else if (!strcmp (key, "spread")) {
            if (sscanf (val, "%lf", &(d_ptr->spread)) != 1) {
                goto error_exit;
            }
        }
        else if (!strcmp (key, "weight")) {
            if (sscanf (val, "%lf", &(d_ptr->weight)) != 1) {
                goto error_exit;
            }
        }
        else if (!strcmp (key, "bragg_curve")) {
            d_ptr->plan->beam->load (val);
        }
		else {
            goto error_exit;
        }
        break;
	}
	return 0;

  error_exit:
    print_and_exit ("Unknown (key,val) combination: (%s,%s)\n", key, val);
    return -1;
}

void
Rt_parms::handle_end_of_section (int section)
{
    switch (section) {
    case 0:
		/* Reading before [PLAN] */
		break;
	case 1:
        /* [PLAN] */
		d_ptr->plan->set_debug(d_ptr->debug);
		d_ptr->plan->set_normalization_dose(d_ptr->dose_norm);
		/* The other parameters are set at the beginning of the dose calculation */
		break;

	case 2:
        /* [BEAM] */
		Rt_beam* new_beam;
		new_beam = new Rt_beam;

		d_ptr->beam_number++; /* initialized to -1, at first run on the beam, beam_number = 0 */
		d_ptr->plan->beam_storage.push_back(new_beam);

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

Rt_plan::Pointer& 
Rt_parms::get_plan ()
{
    return d_ptr->plan;
}

void
Rt_parms::parse_config (
    const char* config_fn
)
{
    /* Read file into string */
    std::ifstream t (config_fn);
    std::stringstream buffer;
    buffer << t.rdbuf();

    std::string buf;
    std::string buf_ori;    /* An extra copy for diagnostics */
    int section = 0;

    std::stringstream ss (buffer.str());

    while (getline (ss, buf)) {
        buf_ori = buf;
        buf = trim (buf);
        buf_ori = trim (buf_ori, "\r\n");

        if (buf == "") continue;
        if (buf[0] == '#') continue;

        if (buf[0] == '[') {
            handle_end_of_section (section);
			if (ci_find (buf, "[PLAN]") != std::string::npos)
			{
				section = 1;
				continue;
			}
			else if (ci_find (buf, "[BEAM]") != std::string::npos)
            {
                section = 2;
                continue;
            }
            else if (ci_find (buf, "[PEAK]") != std::string::npos) 
            {
                section = 3;
                continue;
            }
            else {
                printf ("Parse error: %s\n", buf_ori.c_str());
            }
        }

        size_t key_loc = buf.find ("=");
        if (key_loc == std::string::npos) {
            continue;
        }

        std::string key = buf.substr (0, key_loc);
        std::string val = buf.substr (key_loc+1);
        key = trim (key);
        val = trim (val);

        if (key != "" && val != "") {
            if (this->set_key_val (key.c_str(), val.c_str(), section) < 0) {
                printf ("Parse error: %s\n", buf_ori.c_str());
            }
        }
    }

    handle_end_of_section (section);
}

bool
Rt_parms::parse_args (int argc, char** argv)
{
    int i;
    for (i=1; i<argc; i++) {
        if (argv[i][0] != '-') break;

        if (!strcmp (argv[i], "--debug")) {
            d_ptr->plan->set_debug (true);
        }
        else {
            print_usage ();
            break;
        }
    }

    if (!argv[i]) {
        print_usage ();
    } else {
        this->parse_config (argv[i]);
    }

    if (d_ptr->output_dose_fn == "") {
        fprintf (stderr, "\n** ERROR: Output dose not specified in configuration file!\n");
        return false;
    }

    if (d_ptr->input_ct_fn == "") {
        fprintf (stderr, "\n** ERROR: Patient image not specified in configuration file!\n");
        return false;
    }

    /* load the patient and insert into the plan */
    //Plm_image *ct = plm_image_load (this->input_ct_fn.c_str(), 
    Plm_image::Pointer ct = Plm_image::New (d_ptr->input_ct_fn.c_str(), 
        PLM_IMG_TYPE_ITK_FLOAT);
    if (!ct) {
        fprintf (stderr, "\n** ERROR: Unable to load patient volume.\n");
        return false;
    }

	/* We check if the target prescription or the peaks (SOBP) were set for all the beams  */
	for (int i = 0; i < d_ptr->beam_number; i++)
	{
		if (d_ptr->have_manual_peaks == true && d_ptr->have_prescription == true) {
			fprintf (stderr, "\n** ERROR beam %d: SOBP generation from prescribed distance and manual peaks insertion are incompatible. Please select only one of the two options.\n", d_ptr->beam_number);
			return false;
		}
		if (d_ptr->have_manual_peaks == false && d_ptr->have_prescription == false && d_ptr->target_fn == "") {
			fprintf (stderr, "\n** ERROR beam %d: No prescription made, please use the functions prescription_min & prescription_max, or manually created peaks .\n", d_ptr->beam_number);
			return false;
		}
	}

    d_ptr->plan->set_patient (ct); 
	this->print_verif(d_ptr->plan);

	Volume::Pointer ct_vol = d_ptr->plan->get_patient_volume ();
    Volume::Pointer dose_vol = ct_vol->clone_empty ();

	int dim[3] = {dose_vol->dim[0], dose_vol->dim[1], dose_vol->dim[2]};

    float* total_dose_img = (float*) dose_vol->img;

	for(int i =0; i <= d_ptr->beam_number; i++)
	{
		printf("\nStart dose calculation Beam %d\n", i + 1);
		d_ptr->plan->beam = d_ptr->plan->beam_storage[i];

		/* try to generate plan with the provided parameters */
		if (!d_ptr->plan->init ()) {
			print_and_exit ("ERROR: Unable to initilize plan.\n");
		}

		/* handle auto-generated beam modifiers */
		if (d_ptr->target_fn != "") {
			printf ("Target fn = %s\n", d_ptr->target_fn.c_str());
			d_ptr->plan->set_target (d_ptr->target_fn);
			d_ptr->plan->beam->compute_beam_modifiers ();
			d_ptr->plan->beam->apply_beam_modifiers ();
		}
	
		/* generate depth dose curve, might be manual peaks or 
		optimized based on prescription, or automatic based on target */

		/* Extension of the limits of the PTV - add margins */

		d_ptr->plan->beam->set_proximal_margin (d_ptr->plan->beam->get_proximal_margin());
		/* MDFIX: is it twice the same operation?? */
		d_ptr->plan->beam->set_distal_margin (d_ptr->plan->beam->get_distal_margin());

		if (d_ptr->plan->beam->get_have_manual_peaks() == true && d_ptr->plan->beam->get_have_prescription() == false) {
			/* Manually specified, so do not optimize */
			if (!d_ptr->plan->beam->generate ()) {
				return false;
			}
		} else if (d_ptr->target_fn != "" && !d_ptr->plan->beam->get_have_prescription()) {
			/* Optimize based on target volume */
			Rpl_volume *rpl_vol = d_ptr->plan->beam->rpl_vol;
			d_ptr->plan->beam->set_sobp_prescription_min_max (
				rpl_vol->get_min_wed(), rpl_vol->get_max_wed());
			d_ptr->plan->beam->optimize_sobp ();
		} else {
			/* Optimize based on manually specified range and modulation */
			d_ptr->plan->beam->set_sobp_prescription_min_max (
				d_ptr->plan->beam->get_prescription_min(), d_ptr->plan->beam->get_prescription_max());
			d_ptr->plan->beam->optimize_sobp ();
		}

		/* Generate dose */
		d_ptr->plan->set_debug (true);
		d_ptr->plan->compute_dose ();

		/* Save beam modifiers */
		if (d_ptr->plan->beam->get_aperture_out() != "") {
			Rpl_volume *rpl_vol = d_ptr->plan->beam->rpl_vol;
			Plm_image::Pointer& ap = rpl_vol->get_aperture()->get_aperture_image();
			ap->save_image (d_ptr->plan->beam->get_aperture_out().c_str());
		}

		if (d_ptr->plan->beam->get_range_compensator_out() != "") {
			Rpl_volume *rpl_vol = d_ptr->plan->beam->rpl_vol;
			Plm_image::Pointer& rc = rpl_vol->get_aperture()->get_range_compensator_image();
			rc->save_image (d_ptr->plan->beam->get_range_compensator_out().c_str());
		}

		/* Save projected density volume */
		if (d_ptr->output_proj_img_fn != "") {
			Rpl_volume* proj_img = d_ptr->plan->beam->ct_vol_density;
			if (proj_img) {
				proj_img->save (d_ptr->plan->beam->get_proj_img_out().c_str());
			}
		}

		/* Save projected dose volume */
		if (d_ptr->plan->beam->get_proj_dose_out() != "") {
			Rpl_volume* proj_dose = d_ptr->plan->beam->rpl_dose_vol;
			if (proj_dose) {
				proj_dose->save (d_ptr->plan->beam->get_proj_dose_out().c_str());
			}
		}

		/* Save sigma volume */
		if (d_ptr->plan->beam->get_sigma_out() != "") {
			Rpl_volume* sigma_img = d_ptr->plan->beam->sigma_vol;
			if (sigma_img) {
				sigma_img->save (d_ptr->plan->beam->get_sigma_out().c_str());
			}
		}

		/* Save wed volume */
		if (d_ptr->plan->beam->get_wed_out() != "") {
			Rpl_volume* rpl_vol = d_ptr->plan->beam->rpl_vol;
			if (rpl_vol) {
				rpl_vol->save (d_ptr->plan->beam->get_wed_out().c_str());
			}
		}

		float* beam_dose_img = (float*) d_ptr->plan->beam_storage[i]->get_dose()->get_volume()->img;

		/* Dose cumulation to the plan dose volume */
		for (int j = 0; j < dim[0] * dim[1] * dim[2]; j++)
		{
			total_dose_img[j] += beam_dose_img[j];
		}
	}
	/* Dose max */
	double dose_maxmax =0;
	for (int j = 0; j < dim[0] * dim[1] * dim[2]; j++)
	{
		if (total_dose_img[j] > dose_maxmax)
		{
			dose_maxmax = total_dose_img[j];
		}
	}
	printf("\n dose max: %lg\n", dose_maxmax);

	/* Save dose output */

	Plm_image::Pointer dose = Plm_image::New();
    dose->set_volume (dose_vol);
	d_ptr->plan->set_dose(dose);
	d_ptr->plan->get_dose()->save_image (d_ptr->output_dose_fn.c_str());

    printf ("done.  \n\n");
    return true;
}
