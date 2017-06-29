/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plmdose_config.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "aperture.h"
#include "dose_volume_functions.h"
#include "interpolate.h"
#include "plm_image.h"
#include "plm_math.h"
#include "print_and_exit.h"
#include "proj_matrix.h"
#include "proj_volume.h"
#include "ray_data.h"
#include "ray_trace.h"
#include "rpl_volume.h"
#include "rt_beam.h"
#include "rt_depth_dose.h"
#include "rt_dij.h"
#include "rt_dose.h"
#include "rt_lut.h"
#include "rt_mebs.h"
#include "rt_plan.h"
#include "rt_sigma.h"
#include "string_util.h"
#include "threading.h"
#include "volume.h"

/* Ray Tracer */
double
energy_direct (
    float rgdepth,          /* voxel to dose */
    Rt_beam* beam,
    int beam_idx
)
{
    /* The voxel was not hit directly by the beam */
    if (rgdepth <= 0.0f) {
        return 0.0f;
    }

    /* return the dose at this radiographic depth */
    return (double) beam->get_mebs()->get_depth_dose()[beam_idx]->lookup_energy(rgdepth);
}

void
compute_dose_a (
    Volume::Pointer dose_vol, 
    Rt_beam* beam, 
    const Volume::Pointer ct_vol
)
{
    float* dose_img = (float*) dose_vol->img;

    Aperture::Pointer& ap = beam->get_aperture ();
    Volume *ap_vol = 0;
    const unsigned char *ap_img = 0;
    if (ap->have_aperture_image()) {
        ap_vol = ap->get_aperture_vol ();
        ap_img = ap_vol->get_raw<unsigned char> ();
    }

    /* Dose D(POI) = Dose(z_POI) but z_POI =  rg_comp + depth in CT, 
       if there is a range compensator */
    if (ap->have_range_compensator_image()) {
        add_rcomp_length_to_rpl_volume(beam);
    }

    /* scan through patient CT Volume */
    plm_long ct_ijk[3];
    double ct_xyz[4];
    plm_long idx = 0;
    double idx_ap[2] = {0,0};
    int idx_ap_int[2] = {0,0};
    double rest[2] = {0,0};
    double particle_number = 0;
    float WER = 0;
    float rgdepth = 0;

    for (ct_ijk[2] = 0; ct_ijk[2] < ct_vol->dim[2]; ct_ijk[2]++) {
        for (ct_ijk[1] = 0; ct_ijk[1] < ct_vol->dim[1]; ct_ijk[1]++) {
            for (ct_ijk[0] = 0; ct_ijk[0] < ct_vol->dim[0]; ct_ijk[0]++) {
                double dose = 0.0;

                /* Transform vol index into space coords */
                ct_xyz[0] = (double) (ct_vol->origin[0] + ct_ijk[0] * ct_vol->spacing[0]);
                ct_xyz[1] = (double) (ct_vol->origin[1] + ct_ijk[1] * ct_vol->spacing[1]);
                ct_xyz[2] = (double) (ct_vol->origin[2] + ct_ijk[2] * ct_vol->spacing[2]);
                ct_xyz[3] = (double) 1.0;

                if (beam->get_intersection_with_aperture(idx_ap, idx_ap_int, rest, ct_xyz) == false)
                {
                    printf ("SKIPPING 1\n");
                    continue;
                }

                /* Check that the ray cross the aperture */
                if (idx_ap[0] < 0 || idx_ap[0] > (double) beam->hu_samp_vol->get_proj_volume()->get_image_dim(0)-1
                    || idx_ap[1] < 0 || idx_ap[1] > (double) beam->hu_samp_vol->get_proj_volume()->get_image_dim(1)-1)
                {
//                    printf ("SKIPPING 2\n");
                    continue;
                }

                /* Check that the ray cross the active part of the aperture */
                if (ap_img && beam->is_ray_in_the_aperture(idx_ap_int, ap_img) == false)
                {
                    printf ("SKIPPING 3\n");
                    continue;
                }

                dose = 0;
                rgdepth = beam->rsp_accum_vol->get_rgdepth (ct_xyz);
                WER = compute_PrWER_from_HU (beam->hu_samp_vol->get_rgdepth(ct_xyz));

                const Rt_mebs::Pointer& mebs = beam->get_mebs();
                for (size_t dd_idx = 0; dd_idx < mebs->get_depth_dose().size(); dd_idx++)
                {
                    particle_number = mebs->get_particle_number_xyz (idx_ap_int, rest, dd_idx, beam->get_aperture()->get_dim());
                    if (particle_number != 0 && rgdepth >=0 && rgdepth < mebs->get_depth_dose()[dd_idx]->dend) 
                    {
                        dose += particle_number * WER * energy_direct (rgdepth, beam, dd_idx);
                    }
                }

                /* Insert the dose into the dose volume */
                printf ("Inserting: %f\n", dose);
                idx = volume_index (dose_vol->dim, ct_ijk);
                dose_img[idx] = dose;
            }
        }
    }
}

void
compute_dose_b (
    Rt_beam* beam,
    size_t energy_index,
    const Volume::Pointer ct_vol
)
{
    Rpl_volume *wepl_rv = beam->rsp_accum_vol;
    Volume *wepl_vol = wepl_rv->get_vol();
    float *wepl_img = wepl_vol->get_raw<float> ();

    Rpl_volume *rpl_dose_vol = beam->rpl_dose_vol;
    Volume *dose_vol = rpl_dose_vol->get_vol();
    float *dose_img = dose_vol->get_raw<float> ();

    Rt_mebs::Pointer mebs = beam->get_mebs();
    const Rt_depth_dose *depth_dose = mebs->get_depth_dose()[energy_index];
    std::vector<float>& num_part = mebs->get_num_particles();

    /* scan through rpl volume */
    Aperture::Pointer& ap = beam->get_aperture ();
    Volume *ap_vol = 0;
    const unsigned char *ap_img = 0;
    if (ap->have_aperture_image()) {
        ap_vol = ap->get_aperture_vol ();
        ap_img = ap_vol->get_raw<unsigned char> ();
    }
    const int *dim = wepl_rv->get_image_dim();
    int num_steps = wepl_rv->get_num_steps();
    plm_long ij[2] = {0,0};
    for (ij[1] = 0; ij[1] < dim[1]; ij[1]++) {
        for (ij[0] = 0; ij[0] < dim[0]; ij[0]++) {
            if (ap_img && ap_img[ap_vol->index(ij[0],ij[1],0)] == 0) {
                continue;
            }
            size_t np_index = energy_index * dim[0] * dim[1]
                + ij[1] * dim[0] + ij[0];
            float np = num_part[np_index];
            if (np == 0.f) {
                continue;
            }
            for (int s = 0; s < num_steps; s++) {
                int dose_index = ap_vol->index(ij[0],ij[1],s);
                float wepl = wepl_img[dose_index];
                dose_img[dose_index] += np * depth_dose->lookup_energy(wepl);
            }
        }
    }
}

void
compute_dose_ray_trace_dij_a (
    Rt_beam* beam,
    size_t energy_index,
    const Volume::Pointer ct_vol,
    Volume::Pointer& dose_vol
)
{
    float* dose_img = (float*) dose_vol->img;

    /* Dose D(POI) = Dose(z_POI) but z_POI =  rg_comp + depth in CT, 
       if there is a range compensator */
    if (beam->get_aperture()->have_range_compensator_image())
    {
        add_rcomp_length_to_rpl_volume(beam);
    }

    /* scan through patient CT Volume */
    plm_long ct_ijk[3];
    double ct_xyz[4];
    plm_long idx = 0;
    double idx_ap[2] = {0,0};
    int idx_ap_int[2] = {0,0};
    double rest[2] = {0,0};
    unsigned char* ap_img = (unsigned char*) beam->get_aperture()->get_aperture_volume()->img;
    double particle_number = 0;
    float WER = 0;
    float rgdepth = 0;

    for (ct_ijk[2] = 0; ct_ijk[2] < ct_vol->dim[2]; ct_ijk[2]++) {
        for (ct_ijk[1] = 0; ct_ijk[1] < ct_vol->dim[1]; ct_ijk[1]++) {
            for (ct_ijk[0] = 0; ct_ijk[0] < ct_vol->dim[0]; ct_ijk[0]++) {
                double dose = 0.0;

                /* Transform vol index into space coords */
                ct_xyz[0] = (double) (ct_vol->origin[0] + ct_ijk[0] * ct_vol->spacing[0]);
                ct_xyz[1] = (double) (ct_vol->origin[1] + ct_ijk[1] * ct_vol->spacing[1]);
                ct_xyz[2] = (double) (ct_vol->origin[2] + ct_ijk[2] * ct_vol->spacing[2]);
                ct_xyz[3] = (double) 1.0;

                if (beam->get_intersection_with_aperture(idx_ap, idx_ap_int, rest, ct_xyz) == false)
                {
                    continue;
                }

                /* Check that the ray cross the aperture */
                if (idx_ap[0] < 0 || idx_ap[0] > (double) beam->hu_samp_vol->get_proj_volume()->get_image_dim(0)-1
                    || idx_ap[1] < 0 || idx_ap[1] > (double) beam->hu_samp_vol->get_proj_volume()->get_image_dim(1)-1)
                {
                    continue;
                }

                /* Check that the ray cross the active part of the aperture */
                if (beam->get_aperture()->have_aperture_image() && beam->is_ray_in_the_aperture(idx_ap_int, ap_img) == false)
                {
                    continue;
                }

                dose = 0;
                rgdepth = beam->rsp_accum_vol->get_rgdepth (ct_xyz);
                WER = compute_PrWER_from_HU (beam->hu_samp_vol->get_rgdepth(ct_xyz));

                const Rt_mebs::Pointer& mebs = beam->get_mebs();
                for (size_t dd_idx = 0; dd_idx < mebs->get_depth_dose().size(); dd_idx++)
                {
                    particle_number = mebs->get_particle_number_xyz (idx_ap_int, rest, dd_idx, beam->get_aperture()->get_dim());
                    if (particle_number != 0 && rgdepth >=0 && rgdepth < mebs->get_depth_dose()[dd_idx]->dend) 
                    {
                        dose += particle_number * WER * energy_direct (rgdepth, beam, dd_idx);
                    }
                }

                /* Insert the dose into the dose volume */
                idx = volume_index (dose_vol->dim, ct_ijk);
                dose_img[idx] = dose;
            }
        }
    }
}

void
compute_dose_ray_trace_dij_b (
    Rt_beam* beam,
    size_t energy_index,
    const Volume::Pointer ct_vol,
    Volume::Pointer& dose_vol
)
{
    Rpl_volume *wepl_rv = beam->rsp_accum_vol;
    Volume *wepl_vol = wepl_rv->get_vol();
    float *wepl_img = wepl_vol->get_raw<float> ();

    Rpl_volume *rpl_dose_rv = beam->rpl_dose_vol;
    Volume *rpl_dose_vol = rpl_dose_rv->get_vol();
    float *rpl_dose_img = rpl_dose_vol->get_raw<float> ();

    Rt_mebs::Pointer mebs = beam->get_mebs();
    const Rt_depth_dose *depth_dose = mebs->get_depth_dose()[energy_index];
    std::vector<float>& num_part = mebs->get_num_particles();

    /* Create the beamlet dij matrix */
    Rt_dij rt_dij;

    /* scan through rpl volume */
    Aperture::Pointer& ap = beam->get_aperture ();
    Volume *ap_vol = 0;
    const unsigned char *ap_img = 0;
    if (ap->have_aperture_image()) {
        ap_vol = ap->get_aperture_vol ();
        ap_img = ap_vol->get_raw<unsigned char> ();
    }
    const int *dim = wepl_rv->get_image_dim();
    int num_steps = wepl_rv->get_num_steps();
    plm_long ij[2] = {0,0};
    for (ij[1] = 0; ij[1] < dim[1]; ij[1]++) {
        for (ij[0] = 0; ij[0] < dim[0]; ij[0]++) {
            if (ap_img && ap_img[ap_vol->index(ij[0],ij[1],0)] == 0) {
                continue;
            }
            size_t np_index = energy_index * dim[0] * dim[1]
                + ij[1] * dim[0] + ij[0];
            float np = num_part[np_index];
            if (np == 0.f) {
                continue;
            }
            // Fill in dose
            printf ("[ij] = %d %d\n", ij[1], ij[0]);
            for (int s = 0; s < num_steps; s++) {
                int dose_index = ap_vol->index(ij[0],ij[1],s);
                float wepl = wepl_img[dose_index];
                rpl_dose_img[dose_index] = np * depth_dose->lookup_energy(wepl);
//                printf ("  %f %f %f\n", wepl, np, depth_dose->lookup_energy(wepl));
            }

            // debug
            plm_long nzdv = 0;
            for (plm_long i = 0; i < rpl_dose_vol->npix; i++) {
                if (rpl_dose_img[i] > 0.f) {
                    nzdv ++;
                }
            }
            printf ("nzdv = %d\n", nzdv);

            // Create beamlet dij
            rt_dij.set_from_rpl_dose (
                ij, energy_index, rpl_dose_rv, dose_vol);

            // Write beamlet dij
            // Zero out again
            for (int s = 0; s < num_steps; s++) {
                int dose_index = ap_vol->index(ij[0],ij[1],s);
                rpl_dose_img[dose_index] = 0.f;
            }
        }
    }

    printf ("Dumping...\n");
    if (beam->get_dij_out() != "") {
        rt_dij.dump (beam->get_dij_out());
    }
    printf ("End dumping.\n");
}

void
compute_dose_d (
    Rt_beam* beam,
    size_t energy_index,
    const Volume::Pointer ct_vol
)
{
    beam->get_rt_dose_timing()->timer_dose_calc.resume ();
    Rpl_volume *wepl_rv = beam->rsp_accum_vol;
    Volume *wepl_vol = wepl_rv->get_vol();
    float *wepl_img = wepl_vol->get_raw<float> ();

    Rpl_volume *rpl_dose_vol = beam->rpl_dose_vol;
    Volume *dose_vol = rpl_dose_vol->get_vol();
    float *dose_img = dose_vol->get_raw<float> ();

    Rt_mebs::Pointer mebs = beam->get_mebs();
    const Rt_depth_dose *depth_dose = mebs->get_depth_dose()[energy_index];
    std::vector<float>& num_part = mebs->get_num_particles();
    beam->get_rt_dose_timing()->timer_dose_calc.stop ();

    // Compute sigma for this energy
    beam->get_rt_dose_timing()->timer_sigma.resume ();
    int margins[2] = {0,0};
    float sigma_max = 0;
    compute_sigmas (beam, depth_dose->E0, &sigma_max, "small", margins);
    beam->get_rt_dose_timing()->timer_sigma.stop ();

    beam->get_rt_dose_timing()->timer_dose_calc.resume ();
    Rpl_volume *sigma_rv = beam->sigma_vol;
    Volume *sigma_vol = sigma_rv->get_vol();
    float *sigma_img = sigma_vol->get_raw<float> ();
    const plm_long *sigma_dim = sigma_vol->get_dim();

    // Get the variable magnification at each step
    std::vector <double> lateral_spacing_0 (sigma_dim[2],0);
    std::vector <double> lateral_spacing_1 (sigma_dim[2],0);
    double sid = sigma_rv->get_aperture()->get_distance();
    const double *ap_spacing = sigma_rv->get_aperture()->get_spacing();
    float clipping_dist = sigma_rv->get_front_clipping_plane();
    float step_length = sigma_rv->get_step_length ();
    for (int k = 0; k < sigma_dim[2]; k++) {
        float mag = (sid + clipping_dist + k * step_length) / sid;
        lateral_spacing_0[k] = ap_spacing[0] * mag;
        lateral_spacing_1[k] = ap_spacing[1] * mag;
    }

    // Compute lateral search distance (2.5 max sigma) for each depth
    // (Only needed if pulling dose, not needed if pushing)
    // GCS FIX: This need only be computed once per beam, not once per energy
    std::vector <int> lateral_step_0 (sigma_dim[2],0);
    std::vector <int> lateral_step_1 (sigma_dim[2],0);
    for (int k = 0; k < sigma_dim[2]; k++) {
        float sigma_max = 0.f;
        for (int i = 0; i < sigma_dim[0]*sigma_dim[1]; i++) {
            plm_long idx = k*sigma_dim[0]*sigma_dim[1] + i;
            if (sigma_img[idx] > sigma_max) {
                sigma_max = sigma_img[idx];
            }
        }
        lateral_step_0[k] = ceil (2.5 * sigma_max / lateral_spacing_0[k]);
        lateral_step_1[k] = ceil (2.5 * sigma_max / lateral_spacing_1[k]);
    }

    // Create central axis dose volume
    Rpl_volume *cax_dose_rv = new Rpl_volume;
    if (!cax_dose_rv) return;
    cax_dose_rv->clone_geometry (wepl_rv);
    cax_dose_rv->set_ray_trace_start (RAY_TRACE_START_AT_CLIPPING_PLANE);
    cax_dose_rv->set_aperture (beam->get_aperture());
    cax_dose_rv->set_ct (wepl_rv->get_ct());
    cax_dose_rv->set_ct_limit (wepl_rv->get_ct_limit());
    cax_dose_rv->compute_ray_data();
    cax_dose_rv->set_front_clipping_plane (wepl_rv->get_front_clipping_plane());
    cax_dose_rv->set_back_clipping_plane (wepl_rv->get_back_clipping_plane());
    cax_dose_rv->compute_rpl_void ();
    Volume *cax_dose_vol = cax_dose_rv->get_vol ();
    float *cax_dose_img = cax_dose_vol->get_raw<float> ();
    
    // Compute central axis dose
    Aperture::Pointer& ap = beam->get_aperture ();
    Volume *ap_vol = 0;
    const unsigned char *ap_img = 0;
    if (ap->have_aperture_image()) {
        ap_vol = ap->get_aperture_vol ();
        ap_img = ap_vol->get_raw<unsigned char> ();
    }
    const int *dim = wepl_rv->get_image_dim();
    int num_steps = wepl_rv->get_num_steps();
    plm_long ij[2] = {0,0};
    for (ij[1] = 0; ij[1] < dim[1]; ij[1]++) {
        for (ij[0] = 0; ij[0] < dim[0]; ij[0]++) {
            if (ap_img && ap_img[ap_vol->index(ij[0],ij[1],0)] == 0) {
                continue;
            }
            size_t np_index = energy_index * dim[0] * dim[1]
                + ij[1] * dim[0] + ij[0];
            float np = num_part[np_index];
            if (np == 0.f) {
                continue;
            }
            for (int s = 0; s < num_steps; s++) {
                int dose_index = ap_vol->index(ij[0],ij[1],s);
                float wepl = wepl_img[dose_index];
                cax_dose_img[dose_index] += np * depth_dose->lookup_energy(wepl);
            }
        }
    }

    /* Save sigma volume */
    if (beam->get_sigma_out() != "") {
        std::string fn;
        fn = string_format ("%s/cax-%02d",
            beam->get_sigma_out().c_str(), energy_index);
        cax_dose_rv->save (fn);
        fn = string_format ("%s/sig-%02d", 
            beam->get_sigma_out().c_str(), energy_index);
        sigma_rv->save (fn);
    }
    
    // Smear dose by specified sigma
    for (int s = 0; s < num_steps; s++) {
        double pixel_spacing[2] = {
            lateral_spacing_0[s],
            lateral_spacing_1[s]
        };
        for (ij[1] = 0; ij[1] < dim[1]; ij[1]++) {
            for (ij[0] = 0; ij[0] < dim[0]; ij[0]++) {
                plm_long idx = s*sigma_dim[0]*sigma_dim[1] + ij[1]*dim[0] + ij[0];
                float cax_dose = cax_dose_img[idx];
                if (cax_dose == 0.f) {
                    continue;
                }
                double sigma = (double) sigma_img[idx];
                double sigma_x3 = sigma * 2.5;
                
                // finding the rpl_volume pixels that are contained in the
                // the 3 sigma range
                plm_long ij_min[2], ij_max[2];
                ij_min[0] = ij[0] - ceil (sigma_x3 / lateral_spacing_0[s]);
                if (ij_min[0] < 0) ij_min[0] = 0;
                ij_min[1] = ij[1] - ceil (sigma_x3 / lateral_spacing_1[s]);
                if (ij_min[1] < 0) ij_min[1] = 0;
                ij_max[0] = ij[0] + ceil (sigma_x3 / lateral_spacing_0[s]);
                if (ij_max[0] > dim[0]-1) ij_max[0] = dim[0]-1;
                ij_max[1] = ij[1] + ceil (sigma_x3 / lateral_spacing_1[s]);
                if (ij_max[1] > dim[1]-1) ij_max[1] = dim[1]-1;

                float tot_off_axis = 0.f;
                plm_long ij1[2];
                for (ij1[1] = ij_min[1]; ij1[1] <= ij_max[1]; ij1[1]++) {
                    for (ij1[0] = ij_min[0]; ij1[0] <= ij_max[0]; ij1[0]++) {
                        plm_long idxs = s*sigma_dim[0]*sigma_dim[1]
                            + ij1[1]*dim[0] + ij1[0];
                        double gaussian_center[2] = { 0., 0. };
                        double pixel_center[2] = {
                            (double) ij1[0]-ij[0],
                            (double) ij1[1]-ij[1]
                        };
                        double off_axis_factor;
                        if (sigma == 0)
                        {
                            off_axis_factor = 1;
                        }
                        else
                        {
                            off_axis_factor = double_gaussian_interpolation (
                                gaussian_center, pixel_center,
                                sigma, pixel_spacing);
                        }

                        dose_img[idxs] += cax_dose * off_axis_factor;
#if defined (commentout)
                        // GCS FIX: The below correction would give the
                        // option for dose to tissue
                        / ct_density / STPR;
#endif
                        tot_off_axis += off_axis_factor;
                    }
                }
            }
        }
    }

    // Free temporary memory
    delete cax_dose_rv;
    beam->get_rt_dose_timing()->timer_dose_calc.stop ();
}

void
compute_dose_ray_desplanques (
    Volume* dose_volume, 
    Volume::Pointer ct_vol, 
    Rt_beam* beam, 
    Volume::Pointer final_dose_volume, 
    int beam_index
)
{
    int ijk_idx[3] = {0,0,0};
    int ijk_travel[3] = {0,0,0};
    double xyz_travel[3] = {0.0,0.0,0.0};

    double spacing[3] = { (double) (dose_volume->spacing[0]), (double) (dose_volume->spacing[1]), (double) (dose_volume->spacing[2])};
    int ap_ij[2] = {1,0};
    int dim[2] = {beam->sigma_vol->get_aperture()->get_dim(0),beam->sigma_vol->get_aperture()->get_dim(1)};
    double ray_bev[3] = {0,0,0};
    double xyz_ray_center[3] = {0.0, 0.0, 0.0};
    double entrance_bev[3] = {0.0f, 0.0f, 0.0f}; // coordinates of intersection with the volume in the bev frame
    double xyz_room[3] = {0.0f, 0.0f, 0.0f}; 
    double xyz_room_tmp[3] = {0.0f, 0.0f, 0.0f};
    int ijk_ct[3] = {0,0,0};
    double entrance_length = 0;
    double distance = 0; // distance from the aperture to the POI
    double tmp[3] = {0.0f, 0.0f, 0.0f};
    double ct_density = 0;
    double WER = 0;
    double STPR = 0;
    double sigma = 0;
    int sigma_x3 = 0;
    double rg_length = 0;
    double radius = 0;
    float range_comp = 0;
    float central_axis_dose = 0;
    float off_axis_factor = 0;

    int idx = 0; // index to travel in the dose volume
    int idx_room = 0;
    int i_min = 0;
    int i_max = 0;
    int j_min = 0;
    int j_max = 0;
    bool test = true;
    bool* in = &test;

    float* img = (float*) dose_volume->img;
    float* ct_img = (float*) ct_vol->img;
    float* rc_img = 0;
    unsigned char *ap_img = 0;

    if (beam->get_aperture()->have_range_compensator_image())
    {
        rc_img = (float*) beam->get_aperture()->get_range_compensator_volume ()->img;
    }
	
    if (beam->get_aperture()->have_aperture_image()) {
        Volume::Pointer ap_vol = beam->get_aperture()->get_aperture_volume();
        ap_img = (unsigned char*) ap_vol->img;
    }

    std::vector<float> num_part = beam->get_mebs()->get_num_particles();

    double vec_pdn_tmp[3] = {0,0,0};
    double vec_prt_tmp[3] = {0,0,0};
    double vec_nrm_tmp[3] = {0,0,0};

    vec3_copy(vec_pdn_tmp, beam->rsp_accum_vol->get_proj_volume()->get_incr_c());
    vec3_normalize1(vec_pdn_tmp);
    vec3_copy(vec_prt_tmp, beam->rsp_accum_vol->get_proj_volume()->get_incr_r());
    vec3_normalize1(vec_prt_tmp);
    vec3_copy(vec_nrm_tmp, beam->rsp_accum_vol->get_proj_volume()->get_nrm());
    vec3_normalize1(vec_nrm_tmp);

    for (int i = 0; i < dim[0]*dim[1]; i++)
    {
        if (ap_img[i] == 0 || num_part[beam_index * dim[0] * dim[1] + i] == 0) 
        {
            continue;
        }
        
        Ray_data* ray_data = &beam->sigma_vol->get_Ray_data()[i]; //MD Fix: Why ray_daya->ray for rpl_vol is wrong at this point?

        ap_ij[1] = i / dim[0];
        ap_ij[0] = i- ap_ij[1]*dim[0];
        ray_bev[0] = vec3_dot (ray_data->ray, vec_prt_tmp);
        ray_bev[1] = vec3_dot (ray_data->ray, vec_pdn_tmp);
        ray_bev[2] = -vec3_dot (ray_data->ray, vec_nrm_tmp); // ray_beam_eye_view is already normalized

        /* Calculation of the coordinates of the intersection of the ray with the clipping plane */
        entrance_length = vec3_dist(beam->rsp_accum_vol->get_proj_volume()->get_src(), ray_data->cp);

        vec3_copy(entrance_bev, ray_bev);
        vec3_scale2(entrance_bev, entrance_length);

        if (beam->get_aperture()->have_range_compensator_image())
        {
            range_comp = rc_img[i] * PMMA_DENSITY * PMMA_STPR; // Lucite material: d * rho * WER
        }
        else
        {
            range_comp = 0;
        }
        if (ray_bev[2]  > DRR_BOUNDARY_TOLERANCE)
        {
            for(int k = 0; k < (int) dose_volume->dim[2] ;k++)
            {
                find_xyz_center(xyz_ray_center, ray_bev, dose_volume->origin[2],k, dose_volume->spacing[2]);
                distance = vec3_dist(xyz_ray_center, entrance_bev);
                ct_density = compute_density_from_HU(beam->hu_samp_vol->get_rgdepth(ap_ij, distance));
                STPR = compute_PrSTPR_from_HU(beam->hu_samp_vol->get_rgdepth(ap_ij, distance));
                rg_length = range_comp + beam->rsp_accum_vol->get_rgdepth(ap_ij, distance);
                central_axis_dose = beam->get_mebs()->get_depth_dose()[beam_index]->lookup_energy_integration((float)rg_length, ct_density * dose_volume->spacing[2]) * STPR;
                sigma = beam->sigma_vol->get_rgdepth(ap_ij, distance);
                sigma_x3 = (int) ceil(3 * sigma);

                /* We defined the grid to be updated, the pixels that receive dose from the ray */
                /* We don't check to know if we are still in the matrix because the matrix was build to contain all pixels with a 3 sigma_max margin */
                find_ijk_pixel(ijk_idx, xyz_ray_center, dose_volume);
                i_min = ijk_idx[0] - sigma_x3;
                i_max = ijk_idx[0] + sigma_x3;
                j_min = ijk_idx[1] - sigma_x3;
                j_max = ijk_idx[1] + sigma_x3;
                for (int i2 = i_min; i2 <= i_max; i2++)
                {
                    for (int j2 = j_min; j2 <= j_max; j2++)
                    {
                        if (i2 < 0 || j2 < 0 || i2 >= dose_volume->dim[0] || j2 >= dose_volume->dim[1])
                        {
                            continue;
                        }
                        idx = i2 + (dose_volume->dim[0] * (j2 + dose_volume->dim[1] * k));
                        ijk_travel[0] = i2;
                        ijk_travel[1] = j2;
                        ijk_travel[2] = k;

                        /* calculation of the corresponding position in the room and its HU number*/
                        vec3_copy(xyz_room_tmp, vec_prt_tmp);
                        vec3_scale2(xyz_room_tmp, dose_volume->origin[0] + (float) i2 * dose_volume->spacing[0]);
                        vec3_copy(xyz_room, (xyz_room_tmp));

                        vec3_copy(xyz_room_tmp, vec_pdn_tmp);
                        vec3_scale2(xyz_room_tmp, dose_volume->origin[1] + (float) j2 * dose_volume->spacing[1]);
                        vec3_add2(xyz_room, (xyz_room_tmp));

                        vec3_copy(xyz_room_tmp,  vec_nrm_tmp);
                        vec3_scale2(xyz_room_tmp, (double) (-dose_volume->origin[2] - (float) k * dose_volume->spacing[2]));
                        vec3_add2(xyz_room, (xyz_room_tmp));
                        vec3_add2(xyz_room, beam->rsp_accum_vol->get_proj_volume()->get_src());
						
                        find_ijk_pixel(ijk_ct, xyz_room, ct_vol);
                        idx_room = ijk_ct[0] + (ct_vol->dim[0] * (ijk_ct[1] + ct_vol->dim[1] * ijk_ct[2]));
                        if (ijk_ct[0] < 0 || ijk_ct[1] < 0 || ijk_ct[2] < 0 || ijk_ct[0] >= ct_vol->dim[0] || ijk_ct[1] >= ct_vol->dim[1] || ijk_ct[2] >= ct_vol->dim[2])
                        {
                            WER = PROTON_WER_AIR;
                        }
                        else
                        {
                            WER =  compute_PrWER_from_HU(ct_img[idx_room]);
                        }
                        find_xyz_from_ijk(xyz_travel,dose_volume,ijk_travel);
                        radius = vec3_dist(xyz_travel,xyz_ray_center); 
                        if (sigma == 0)
                        {
                            off_axis_factor = 1;
                        }
                        else if (radius > sqrt(0.25 * spacing[0] * spacing [0] + 0.25 * spacing[1] * spacing[1]) + 3 * sigma )
                        {
                            off_axis_factor = 0;
                        }
                        else
                        {
                            off_axis_factor = double_gaussian_interpolation(xyz_ray_center, xyz_travel,sigma, spacing);
                        }
                        /* SOBP is weighted by the weight of the pristine peak */
                        img[idx] += num_part[beam_index * dim[0] * dim[1] + i] * central_axis_dose 
                            * WER // dose = dose_w * WER
                            * off_axis_factor ;
                    }			
                }
            }
        }
    }

    float* final_dose_img = (float*) final_dose_volume->img;
    int ijk[3] = {0,0,0};
    float ijk_bev[3] = {0,0,0};
    int ijk_bev_trunk[3];
    float xyz_bev[3] = {0.0,0.0,0.0};
    plm_long mijk_f[3];
    plm_long mijk_r[3];
    plm_long idx_lower_left = 0;
    float li_frac1[3];
    float li_frac2[3];
    const plm_long *dim_ct = ct_vol->dim;
    plm_long dose_bev_dim[3] = { dose_volume->dim[0], dose_volume->dim[1], dose_volume->dim[2]};

    for (ijk[0] = 0; ijk[0] < dim_ct[0]; ijk[0]++)
    {
        for (ijk[1] = 0; ijk[1] < dim_ct[1]; ijk[1]++)
        {
            for (ijk[2] = 0; ijk[2] < dim_ct[2]; ijk[2]++)
            {
                idx = ijk[0] + dim_ct[0] *(ijk[1] + ijk[2] * dim_ct[1]);
                if ( ct_img[idx] >= -1000) // in air we have no dose, we let the voxel number at 0!
                {   
                    final_dose_volume->get_xyz_from_ijk(xyz_room, ijk);

                    /* xyz contains the coordinates of the pixel in the room coordinates */
                    /* we now calculate the coordinates of this pixel in the dose_volume coordinates */

                    vec3_sub3(tmp,  beam->rsp_accum_vol->get_proj_volume()->get_src(), xyz_room);
                    xyz_bev[0] = (float) -vec3_dot(tmp, vec_prt_tmp);
                    xyz_bev[1] = (float) -vec3_dot(tmp,  vec_pdn_tmp);
                    xyz_bev[2] = (float) vec3_dot(tmp,  vec_nrm_tmp);
                    dose_volume->get_ijk_from_xyz(ijk_bev, xyz_bev, in);

                    if (*in == true)
                    {
                        dose_volume->get_ijk_from_xyz(ijk_bev_trunk, xyz_bev, in);
                        li_clamp_3d(ijk_bev, mijk_f, mijk_r, li_frac1, li_frac2, dose_volume);
                        idx_lower_left =  mijk_f[0] + dose_bev_dim[0] *(mijk_f[1] + mijk_f[2] * dose_bev_dim[1]);
                        final_dose_img[idx] += li_value(li_frac1[0], li_frac2[0], li_frac1[1], li_frac2[1], li_frac1[2], li_frac2[2], idx_lower_left, img, dose_volume);
                    }
                }
            }   
        }     
    }
    return;
}

void 
compute_dose_ray_sharp (
    const Volume::Pointer ct_vol, 
    Rt_beam* beam, 
    Rpl_volume* rpl_dose_volume,  
    int beam_index,
    const int* margins
)
{
    int ap_ij_lg[2] = {0,0};
    int ap_ij_sm[2] = {0,0};
    int dim_lg[3] = {0,0,0};
    int dim_sm[3] = {0,0,0};
    int idx2d_sm = 0;
    int idx2d_lg = 0;
    int idx3d_sm = 0;
    int idx3d_lg = 0;
    int idx3d_travel = 0;
    int i_min = 0;
    int i_max = 0;
    int j_min = 0;
    int j_max = 0;

    float ct_density = 0;
    float STPR = 0;
    double sigma = 0;
    double sigma_x3 = 0;
    double rg_length = 0;
    float central_axis_dose = 0;
    float off_axis_factor = 0;
    double minimal_lateral = 0;
    double lateral_step[2] = {0,0};
    double central_ray_xyz[3] = {0.0, 0.0, 0.0};
    double travel_ray_xyz[3] = {0.0, 0.0, 0.0};

    dim_lg[0] = rpl_dose_volume->get_vol()->dim[0];
    dim_lg[1] = rpl_dose_volume->get_vol()->dim[1];
    dim_lg[2] = rpl_dose_volume->get_vol()->dim[2];
    dim_sm[0] = beam->rsp_accum_vol->get_vol()->dim[0];
    dim_sm[1] = beam->rsp_accum_vol->get_vol()->dim[1];
    dim_sm[2] = beam->rsp_accum_vol->get_vol()->dim[2];
	
    float* rpl_img = (float*) beam->rsp_accum_vol->get_vol()->img;
    float* sigma_img = (float*) beam->sigma_vol->get_vol()->img;
    float* rpl_dose_img = (float*) rpl_dose_volume->get_vol()->img;
    float* ct_rpl_img = (float*) beam->hu_samp_vol->get_vol()->img;
    float* rc_img = 0;
    unsigned char *ap_img = 0;
    float range_comp = 0;

    if (beam->get_aperture()->have_aperture_image()) {
        Volume::Pointer ap_vol = beam->get_aperture()->get_aperture_volume();
        ap_img = (unsigned char*) ap_vol->img;
    }

    if (beam->get_aperture()->have_range_compensator_image())
    {
        rc_img = (float*) beam->get_aperture()->get_range_compensator_volume ()->img;
    }

    /* Creation of the rpl_volume containing the coordinates xyz (beam eye view) and the CT density vol*/
    std::vector<double> xyz_init (4,0);
    std::vector< std::vector<double> > xyz_coor_vol (dim_lg[0]*dim_lg[1]*dim_lg[2], xyz_init);
    calculate_rpl_coordinates_xyz (&xyz_coor_vol, rpl_dose_volume);

    for (int m = 0; m < dim_lg[0] * dim_lg[1] * dim_lg[2]; m++)
    {
        rpl_dose_img[m] = 0;
    }

    /* calculation of the lateral steps in which the dose is searched constant with depth */
    std::vector <double> lateral_minimal_step (dim_lg[2],0);
    std::vector <double> lateral_step_x (dim_lg[2],0);
    std::vector <double> lateral_step_y (dim_lg[2],0);

    minimal_lateral = beam->get_aperture()->get_spacing(0);
    if (minimal_lateral < beam->get_aperture()->get_spacing(1))
    {
        minimal_lateral = beam->get_aperture()->get_spacing(1);
    }

    for (int k = 0; k < dim_sm[2]; k++)
    {
        lateral_minimal_step[k] = (beam->rsp_accum_vol->get_front_clipping_plane() + beam->rsp_accum_vol->get_aperture()->get_distance() + (double) k) * minimal_lateral / beam->rsp_accum_vol->get_aperture()->get_distance();
        lateral_step_x[k] = (beam->rsp_accum_vol->get_front_clipping_plane() + beam->rsp_accum_vol->get_aperture()->get_distance() + (double) k) * beam->get_aperture()->get_spacing(0) / beam->rsp_accum_vol->get_aperture()->get_distance();
        lateral_step_y[k] = (beam->rsp_accum_vol->get_front_clipping_plane() + beam->rsp_accum_vol->get_aperture()->get_distance() + (double) k) *beam->get_aperture()->get_spacing(1) / beam->rsp_accum_vol->get_aperture()->get_distance();
    }

    std::vector<float>& num_part = beam->get_mebs()->get_num_particles();

    printf ("Aperture margin = %d %d\n", margins[0], margins[1]);
    
    /* calculation of the dose in the rpl_volume */
    for (ap_ij_lg[0] = margins[0];
         ap_ij_lg[0] < rpl_dose_volume->get_vol()->dim[0]-margins[0];
         ap_ij_lg[0]++)
    {
        for (ap_ij_lg[1] = margins[1];
             ap_ij_lg[1] < rpl_dose_volume->get_vol()->dim[1]-margins[1];
             ap_ij_lg[1]++)
        {
            bool debug = false;
//            if (ap_ij_lg[0] == 28 && ap_ij_lg[1] == 28) {
//                debug = true;
//            }
            
            ap_ij_sm[0] = ap_ij_lg[0] - margins[0];
            ap_ij_sm[1] = ap_ij_lg[1] - margins[1];
            idx2d_lg = ap_ij_lg[1] * dim_lg[0] + ap_ij_lg[0];
            idx2d_sm = ap_ij_sm[1] * dim_sm[0] + ap_ij_sm[0];

            if (beam->get_aperture()->have_aperture_image())
            {
                if ((float) ap_img[idx2d_sm] == 0 || num_part[beam_index * beam->get_aperture()->get_dim(0) * beam->get_aperture()->get_dim(1) + idx2d_sm] == 0)
                {
                    continue;
                }
            }
            if (beam->get_aperture()->have_range_compensator_image())
            {
                // Lucite Material: d * rho * WER, MD Fix
                range_comp = rc_img[idx2d_sm] * PMMA_DENSITY * PMMA_STPR;
            }
            else
            {
                range_comp = 0;
            }

            for (int k = 0; k < dim_sm[2]; k++)
            {
                idx3d_lg = idx2d_lg + k * dim_lg[0]*dim_lg[1];
                idx3d_sm = idx2d_sm + k * dim_sm[0]*dim_sm[1];

                central_ray_xyz[0] = xyz_coor_vol[idx3d_lg][0];
                central_ray_xyz[1] = xyz_coor_vol[idx3d_lg][1];
                central_ray_xyz[2] = xyz_coor_vol[idx3d_lg][2];

                lateral_step[0] = lateral_step_x[k];
                lateral_step[1] = lateral_step_x[k];

                ct_density = compute_density_from_HU(ct_rpl_img[idx3d_sm]);
                STPR = compute_PrSTPR_from_HU(ct_rpl_img[idx3d_sm]);

                rg_length = range_comp + rpl_img[idx3d_sm];
                central_axis_dose = num_part[beam_index * beam->get_aperture()->get_dim(0)* beam->get_aperture()->get_dim(1) + idx2d_sm] * beam->get_mebs()->get_depth_dose()[beam_index]->lookup_energy_integration(rg_length, ct_density * beam->rsp_accum_vol->get_vol()->spacing[2]) * STPR;

                if (central_axis_dose <= 0) // no dose on the axis, no dose scattered
                {
                    continue;
                }

                sigma = (double) sigma_img[idx3d_sm];
                sigma_x3 = sigma * 3;

                /* finding the rpl_volume pixels that are contained in the the 3 sigma range */                    
                i_min = ap_ij_lg[0] - (int) ceil(sigma_x3 / lateral_minimal_step[k]);
                if (i_min < 0 ) {i_min = 0;}
                i_max = ap_ij_lg[0] + (int) ceil(sigma_x3 / lateral_minimal_step[k]);
                if (i_max > dim_lg[0]-1 ) {i_max = dim_lg[0]-1;}
                j_min = ap_ij_lg[1] - (int) ceil(sigma_x3 / lateral_minimal_step[k]);
                if (j_min < 0 ) {j_min = 0;}
                j_max = ap_ij_lg[1] + (int) ceil(sigma_x3 / lateral_minimal_step[k]);
                if (j_max > dim_lg[1]-1 ) {j_max = dim_lg[1]-1;}

                float tot_off_axis = 0.f;
                for (int i1 = i_min; i1 <= i_max; i1++) {
                    for (int j1 = j_min; j1 <= j_max; j1++) {
                        idx3d_travel = k * dim_lg[0]*dim_lg[1] + j1 * dim_lg[0] + i1;

                        travel_ray_xyz[0] = xyz_coor_vol[idx3d_travel][0];
                        travel_ray_xyz[1] = xyz_coor_vol[idx3d_travel][1];
                        travel_ray_xyz[2] = xyz_coor_vol[idx3d_travel][2];

                        if (sigma == 0)
                        {
                            off_axis_factor = 1;
                        }
                        else
                        {
                            off_axis_factor = double_gaussian_interpolation (
                                central_ray_xyz, travel_ray_xyz,
                                sigma, lateral_step);
                        }

                        rpl_dose_img[idx3d_travel] += central_axis_dose 
                            * off_axis_factor / ct_density / STPR;
                        tot_off_axis += off_axis_factor;
                        
                    } //for j1
                } //for i1
                if (debug) {
                    printf ("%d %f %f %f %f\n", k, (float) central_axis_dose,
                        tot_off_axis, ct_density, STPR);
                }
            } // for k
        } // ap_ij[1]
    } // ap_ij[0]   
}

void compute_dose_ray_shackleford (
    Volume::Pointer dose_vol,
    Rt_plan* plan,
    Rt_beam* beam,
    int beam_index,
    std::vector<double>* area,
    std::vector<double>* xy_grid,
    int radius_sample,
    int theta_sample)
{
    int ijk[3] = {0,0,0};
    double xyz[4] = {0,0,0,1};
    double xyz_travel[4] = {0,0,0,1};
    double tmp_xy[4] = {0,0,0,1};
    double tmp_cst = 0;
    int idx = 0;
    const plm_long *dose_dim = dose_vol->dim;
    double vec_ud[4] = {0,0,0,1};
    double vec_rl[4] = {0,0,0,1};
    float* dose_img = (float*) dose_vol->img;
    double sigma_travel = 0;
    double sigma_3 = 0;
    double rg_length = 0;
    float ct_density = 0;
    float STPR = 0;
    float HU = 0;
    double central_sector_dose = 0;
    double radius = 0;
    double dr = 0;

    double idx_ap[2] = {0,0};
    int idx_ap_int[2] = {0,0};
    double rest[2] = {0,0};
    float particle_number = 0;

    unsigned char *ap_img = 0;	
    if (beam->get_aperture()->have_aperture_image()) {
        Volume::Pointer ap_vol = beam->get_aperture()->get_aperture_volume();
        ap_img = (unsigned char*) ap_vol->img;
    }

    /* Dose D(POI) = Dose(z_POI) but z_POI =  rg_comp + depth in CT, if there is a range compensator */
    if (beam->get_aperture()->have_range_compensator_image())
    {
        add_rcomp_length_to_rpl_volume(beam);
    }
    vec3_copy(vec_ud, beam->rsp_accum_vol->get_proj_volume()->get_incr_c());
    vec3_normalize1(vec_ud);
    vec3_copy(vec_rl, beam->rsp_accum_vol->get_proj_volume()->get_incr_r());
    vec3_normalize1(vec_rl);

    for (ijk[0] = 0; ijk[0] < dose_dim[0]; ijk[0]++){
        printf("%d ", ijk[0]);
        for (ijk[1] = 0; ijk[1] < dose_dim[1]; ijk[1]++){
            for (ijk[2] = 0; ijk[2] < dose_dim[2]; ijk[2]++){
                idx = ijk[0] + dose_dim[0] * (ijk[1] + dose_dim[1] * ijk[2]);

                /* calculation of the pixel coordinates in the room coordinates */
                xyz[0] = (double) dose_vol->origin[0] + ijk[0] * dose_vol->spacing[0];
                xyz[1] = (double) dose_vol->origin[1] + ijk[1] * dose_vol->spacing[1];
                xyz[2] = (double) dose_vol->origin[2] + ijk[2] * dose_vol->spacing[2]; // xyz[3] always = 1.0
                sigma_3 = 3 * beam->sigma_vol_lg->get_rgdepth(xyz);

                for (int i = 0; i < radius_sample; i++)
                {
                    for (int j =0; j < theta_sample; j++)
                    {
                        vec3_copy(xyz_travel, xyz);

                        /* calculation of the center of the sector */
                        vec3_copy(tmp_xy, vec_ud);
                        tmp_cst = (double) (*xy_grid)[2*(i*theta_sample+j)] * sigma_3; // xy_grid is normalized to a circle of radius sigma x 3 = 1
                        vec3_scale2(tmp_xy, tmp_cst);
                        vec3_add2(xyz_travel,tmp_xy);

                        vec3_copy(tmp_xy, vec_rl);
                        tmp_cst = (double) (*xy_grid)[2*(i*theta_sample+j)+1] * sigma_3;
                        vec3_scale2(tmp_xy, tmp_cst);
                        vec3_add2(xyz_travel,tmp_xy);
							
                        rg_length = beam->rsp_accum_vol->get_rgdepth(xyz_travel);
                        HU = beam->rpl_vol_samp_lg->get_rgdepth(xyz_travel);
                        if (beam->get_intersection_with_aperture(idx_ap, idx_ap_int, rest, xyz_travel) == false)
                        {
                            continue;
                        }

                        /* Check that the ray cross the aperture */
                        if (idx_ap[0] < 0 || idx_ap[0] > (double) beam->hu_samp_vol->get_proj_volume()->get_image_dim(0)-1
                            || idx_ap[1] < 0 || idx_ap[1] > (double) beam->hu_samp_vol->get_proj_volume()->get_image_dim(1)-1)
                        {
                            continue;
                        }
                        /* Check that the ray cross the active part of the aperture */
                        if (beam->get_aperture()->have_aperture_image() && beam->is_ray_in_the_aperture(idx_ap_int, ap_img) == false)
                        {
                            continue;
                        }
                        /* Check that the spot map is positive for this ray */
                        particle_number = beam->get_mebs()->get_particle_number_xyz(idx_ap_int, rest, beam_index, beam->get_aperture()->get_dim());
                        if (particle_number <= 0)
                        {
                            continue;
                        }
                        ct_density = compute_density_from_HU(HU);
                        STPR = compute_PrSTPR_from_HU(HU);
							
                        if (rg_length <= 0)
                        {
                            continue;
                        }
                        else
                        {
		
                            /* the dose from that sector is summed */
                            sigma_travel = beam->sigma_vol->get_rgdepth(xyz_travel);
                            radius = vec3_dist(xyz, xyz_travel);
								
                            if (sigma_travel < radius / 3)
                            {
                                continue;
                            }
                            else
                            {
                                central_sector_dose = particle_number * beam->get_mebs()->get_depth_dose()[beam_index]->lookup_energy_integration((float) rg_length, ct_density * beam->rsp_accum_vol->get_vol()->spacing[2])* STPR * (1/(sigma_travel*sqrt(2*M_PI)));
                                dr = sigma_3 / (2* radius_sample);
                                dose_img[idx] +=  
                                    central_sector_dose
                                    * compute_PrWER_from_HU(HU)
                                    * get_off_axis(radius, dr, sigma_3/3) 
                                    * beam->get_mebs()->get_weight()[beam_index]; 
                            }
                        }
                    }
                }
            }
        }
    }
}

void add_rcomp_length_to_rpl_volume (Rt_beam* beam)
{
    const plm_long *dim = beam->rsp_accum_vol->get_vol()->dim;
    float* rpl_img = (float*) beam->rsp_accum_vol->get_vol()->img;
    float* rc_img = (float*) beam->rsp_accum_vol->get_aperture()->get_range_compensator_volume()->img;
    int idx = 0;

    for(int i = 0; i < dim[0] * dim[1]; i++)
    {
        for (int k = 0; k < dim[2]; k++)
        {
            idx = i + k * dim[0] * dim[1];
            rpl_img[idx] += rc_img[i] * PMMA_DENSITY*PMMA_STPR; // Lucite material : d * rho * WER
        }
    }
}
