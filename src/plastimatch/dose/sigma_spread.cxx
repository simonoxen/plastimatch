#include "math.h"

#include "interpolate.h"
#include "ion_beam.h"
#include "ion_dose.h"
#include "photon_beam.h"
#include "photon_dose.h"
#include "sigma_spread.h"
#include "rpl_volume.h"
#include "ray_data.h"
#include "ray_trace.h"

void 
convert_radiologic_length_to_sigma (
    Ion_plan* ion_plan, 
    float energy, 
    float* sigma_max
)
{
    /* Now we only have a rpl_volume without compensator, from which we need to compute the sigma along this ray */
    /* we extract a ray, we apply the sigma_function given the y0 according to the Hong algorithm and we put it back in the volum */
    /* at the end we have transformed our rpl_volume (not cumulative) in a sigma (in reality y0) volume */

    float *sigma_img = (float*) ion_plan->sigma_vol->get_vol()->img;
    float *ct_img = (float*) ion_plan->ct_vol_density->get_vol()->img;
    float *ct_rglength = (float*) ion_plan->rpl_vol->get_vol()->img;

    plm_long ires[3] = {
        ion_plan->sigma_vol->get_vol()->dim[0], 
        ion_plan->sigma_vol->get_vol()->dim[1], 
        ion_plan->sigma_vol->get_vol()->dim[2]
    };
    std::vector<float> french_fries_sigma (ires[2],0);
    std::vector<float> french_fries_density (ires[2],0);

    for (int apert_idx = 0; apert_idx < ires[0]*ires[1]; apert_idx++)
    {   
        for (int s = 0; s < ires[2]; s++)
        {
            french_fries_sigma[s] = ct_rglength[ires[0]*ires[1]*s + apert_idx]; // the sigma fries is initialized with density
            french_fries_density[s] = ct_img[ires[0]*ires[1]*s + apert_idx];
        }

        length_to_sigma (&french_fries_sigma,&french_fries_density, ion_plan->sigma_vol->get_vol()->spacing[2], sigma_max, energy, ion_plan->beam->get_source_size());

#if defined (commentout)
        for (std::vector<float>::iterator it = french_fries_sigma.begin();
             it != french_fries_sigma.end(); it++)
        {
            printf ("PS %g\n", *it);
        }
#endif

        for (int s = 0; s < ires[2]; s++)
        {
            sigma_img[ires[0]*ires[1]*s + apert_idx] = french_fries_sigma[s];
        }
    }
    printf("sigma_max = %lg\n", *sigma_max);
}

void convert_radiologic_length_to_sigma(Photon_plan* ion_plan, float energy, float* sigma_max)
{
    /* Now we only have a rpl_volume without compensator, from which we need to compute the sigma along this ray */
    /* we extract a ray, we apply the sigma_function given the y0 according to the Hong algorithm and we put it back in the volum */
    /* at the end we have transformed our rpl_volume (not cumulative) in a sigma (in reality y0) volume */

    float *sigma_img = (float*) ion_plan->sigma_vol->get_vol()->img;
    float *ct_img = (float*) ion_plan->ct_vol_density->get_vol()->img;
    float *ct_rglength = (float*) ion_plan->rpl_vol->get_vol()->img;

    plm_long ires[3] = {
        ion_plan->sigma_vol->get_vol()->dim[0], 
        ion_plan->sigma_vol->get_vol()->dim[1], 
        ion_plan->sigma_vol->get_vol()->dim[2]};

    std::vector<float> french_fries_sigma (ires[2],0);
    std::vector<float> french_fries_density (ires[2],0);

    for (int apert_idx = 0; apert_idx < ires[0]*ires[1]; apert_idx++)
    {   
        for (int s = 0; s < ires[2]; s++)
        {
            french_fries_sigma[s] = ct_rglength[ires[0]*ires[1]*s + apert_idx]; // the sigma fries is initialized with density
            french_fries_density[s] = ct_img[ires[0]*ires[1]*s + apert_idx];
        }

        length_to_sigma_photon(&french_fries_sigma,&french_fries_density, ion_plan->sigma_vol->get_vol()->spacing[2], sigma_max, energy, ion_plan->get_source_size());

        for (int s = 0; s < ires[2]; s++)
        {
            sigma_img[ires[0]*ires[1]*s + apert_idx] = french_fries_sigma[s];
        }
    }
    printf("sigma_max = %lg\n", *sigma_max);
}

void convert_radiologic_length_to_sigma_lg(Ion_plan* ion_plan, float energy, float* sigma_max) //Rpl_volume* sigma_vol, Rpl_volume* ct_vol, float energy, float spacing_z, float* sigma_max)
{
    /* Now we only have a rpl_volume without compensator, from which we need to compute the sigma along this ray */
    /* we extract a ray, we apply the sigma_function given the y0 according to the Hong algorithm and we put it back in the volum */
    /* at the end we have transformed our rpl_volume (not cumulative) in a sigma (in reality y0) volume */

    float *sigma_img = (float*) ion_plan->sigma_vol_lg->get_vol()->img;
    float *ct_img = (float*) ion_plan->ct_vol_density_lg->get_vol()->img;
    float *ct_rglength = (float*) ion_plan->rpl_vol_lg->get_vol()->img;

    plm_long ires[3] = {
        ion_plan->sigma_vol_lg->get_vol()->dim[0], 
        ion_plan->sigma_vol_lg->get_vol()->dim[1], 
        ion_plan->sigma_vol_lg->get_vol()->dim[2]
    };

    std::vector<float> french_fries_sigma (ires[2],0);
    std::vector<float> french_fries_density (ires[2],0);

    for (int apert_idx = 0; apert_idx < ires[0]*ires[1]; apert_idx++)
    {   
        for (int s = 0; s < ires[2]; s++)
        {
            french_fries_sigma[s] = ct_rglength[ires[0]*ires[1]*s + apert_idx]; // the sigma fries is initialized with density
            french_fries_density[s] = ct_img[ires[0]*ires[1]*s + apert_idx];
        }

        length_to_sigma(&french_fries_sigma,&french_fries_density, ion_plan->sigma_vol_lg->get_vol()->spacing[2], sigma_max, energy, ion_plan->beam->get_source_size());
        for (int s = 0; s < ires[2]; s++)
        {
            sigma_img[ires[0]*ires[1]*s + apert_idx] = french_fries_sigma[s];
        }
    }
    printf("new sigma_max = %lg\n", *sigma_max);
}

void convert_radiologic_length_to_sigma_lg(Photon_plan* ion_plan, float energy, float* sigma_max) //Rpl_volume* sigma_vol, Rpl_volume* ct_vol, float energy, float spacing_z, float* sigma_max)
{
    /* Now we only have a rpl_volume without compensator, from which we need to compute the sigma along this ray */
    /* we extract a ray, we apply the sigma_function given the y0 according to the Hong algorithm and we put it back in the volum */
    /* at the end we have transformed our rpl_volume (not cumulative) in a sigma (in reality y0) volume */

    float *sigma_img = (float*) ion_plan->sigma_vol_lg->get_vol()->img;
	float *ct_img = (float*) ion_plan->ct_vol_density_lg->get_vol()->img;
	float *ct_rglength = (float*) ion_plan->rpl_vol_lg->get_vol()->img;

    int ires[3] = {ion_plan->sigma_vol_lg->get_vol()->dim[0], ion_plan->sigma_vol_lg->get_vol()->dim[1], ion_plan->sigma_vol_lg->get_vol()->dim[2]};

    std::vector<float> french_fries_sigma (ires[2],0);
    std::vector<float> french_fries_density (ires[2],0);

    for (int apert_idx = 0; apert_idx < ires[0]*ires[1]; apert_idx++)
    {   
        for (int s = 0; s < ires[2]; s++)
        {
            french_fries_sigma[s] = ct_rglength[ires[0]*ires[1]*s + apert_idx]; // the sigma fries is initialized with density
            french_fries_density[s] = ct_img[ires[0]*ires[1]*s + apert_idx];
        }

		length_to_sigma_photon(&french_fries_sigma,&french_fries_density, ion_plan->sigma_vol_lg->get_vol()->spacing[2], sigma_max, energy, ion_plan->get_source_size());

        for (int s = 0; s < ires[2]; s++)
        {
            sigma_img[ires[0]*ires[1]*s + apert_idx] = french_fries_sigma[s];
        }
    }
    printf("new sigma_max = %lg\n", *sigma_max);
}

void 
length_to_sigma (
    std::vector<float>* p_sigma, 
    const std::vector<float>* p_density, 
    float spacing_z, 
    float* sigma_max, 
    float energy, 
    float sourcesize
)
{
    std::vector<float> tmp_rglength (p_sigma->size(), 0);
    int first_non_null_loc = 0;
    spacing_z = spacing_z/10; // converted to cm (the Highland formula is in cm!)

    /* initializiation of all the french_fries, except sigma, which is the output and calculate later */
    for(int i = 0; i < (int) p_sigma->size();i++)
    {
        if(i == 0)
        {
            tmp_rglength[i] = (*p_sigma)[i]; // remember that at this point french_fries_sigma is only a rglngth function without compensator
            (*p_sigma)[i] = 0;
        } 
        else 
        {
            tmp_rglength[i] = (*p_sigma)[i]-(*p_sigma)[i-1]; // rglength in the pixel
            (*p_sigma)[i] = 0;
        }
    }

    //We can now compute the sigma french_fries!!!!

    /* Step 1: the sigma is filled with zeros, so we let them at 0 as long as rg_length is 0, meaning the ray is out of the volume */
    /* we mark the first pixel in the volume, and the calculations will start with this one */
    for (int i = 0; i < (int) p_sigma->size(); i++)
    {
        if (tmp_rglength[i] > 0)
        {
            first_non_null_loc = i;
            break;
        }
        if (i == p_sigma->size())
        {
            first_non_null_loc = p_sigma->size()-1;
            printf("\n the french_fries is completely zeroed, the ray seems to not intersect the volume\n");
            return;
        }
    }

    /* Step 2: Each pixel in the volume will receive its sigma (in reality y0) value, according to the differential Highland formula */

    float energy_callback = energy;
    float mc2 = 939.4f;          /* proton mass at rest (MeV) */
    float c = 299792458.0f;        /* speed of light (m/s2) */

    float p = 0.0;              /* Proton momentum (passed in)          */
    float v = 0.0;              /* Proton velocity (passed in)          */
    float stop = 0;		/* stopping power energy (MeV.cm2/g) */
	
    float sum = 0.0;		/* integration expressions, right part of equation */
    float function_to_be_integrated; /* right term to be integrated in the Highland equation */
    float inverse_rad_length_integrated = 0; /* and left part */
    float y0 = 0;               /* y0 value used to update the french_fries values */
    float step;                 /* step of integration, will depends on the radiologic length */

    float POI_depth;            /* depth of the point of interest (where is calculated the sigma value)in cm - centered at the pixel center*/
    float pixel_depth;          /* depth of the contributing pixel to total sigma (in cm) - center between 2 pixels, the difference in rglength comes from the center of the previous pixel to the center of this pixel*/
    double sigma_source;
    double sigma_range_compensator;
    double sigma_patient;

    std::vector<double> pv_cache (p_sigma->size(), 0);
    std::vector<double> inv_rad_len (p_sigma->size(), 0);
    std::vector<double> stop_cache (p_sigma->size(), 0);
    for (int i = first_non_null_loc; i < (int) p_sigma->size(); i++)
    {
        p = sqrt(2*energy*mc2+energy*energy)/c; // in MeV.s.m-1
        v = c*sqrt(1-pow((mc2/(energy+mc2)),2)); //in m.s-1
        pv_cache[i] = p * v;
        inv_rad_len[i] = 1.0f / LR_interpolation((*p_density)[i]);
        stop_cache[i] = getstop (energy) * WER_interpolation((*p_density)[i]) * (*p_density)[i]; // dE/dx_mat = dE /dx_watter * WER * density (lut in g/cm2)

        sum = 0;
        inverse_rad_length_integrated = 0;

        POI_depth = (float) (i+0.5)*spacing_z;

        /*integration */
        energy = energy_callback;
        for (int j = first_non_null_loc; j <= i && energy > 0;j++)
        {
            if (i == j)
            {
                pixel_depth = (j+.25f)*spacing_z; // we integrate only up to the voxel center, not the whole pixel
                step = spacing_z/2;
            }
            else
            {
                pixel_depth = (j+0.5f)*spacing_z;
                step = spacing_z;
            }
            
            function_to_be_integrated = pow(((POI_depth - pixel_depth)/pv_cache[j]),2) * inv_rad_len[j]; //i in cm
            sum += function_to_be_integrated*step;

            inverse_rad_length_integrated += step * inv_rad_len[j];

            /* energy is updated after passing through dz */
            energy = energy - step * stop_cache[j];
        }

        if (energy  <= 0) // sigma formula is not defined anymore
        {
            return; // we can exit as the rest of the french_fries_sigma equals already 0
        }
        
        // We have reached the POI pixel and we can store the y0 value
        sigma_source = sourcesize * ((20 + POI_depth) / 187);
        sigma_range_compensator = 0.331 * 0.00313 * (20 + POI_depth + 4.4); // 4.4 is the fraction of the RC - effective scattering depth on 13cm
        sigma_patient = 14.10f *(1.0f+1.0f/9.0f*log10(inverse_rad_length_integrated))* (float) sqrt(sum); // in cm
        (*p_sigma)[i] = 10 * ( //*10 because sigma is used later in mm
            sqrt(sigma_source * sigma_source + sigma_range_compensator * sigma_range_compensator + sigma_patient * sigma_patient));

        if (*sigma_max < (*p_sigma)[i])
        {
            *sigma_max = (*p_sigma)[i];
        }
    }
}

void 
length_to_sigma_slow (
    std::vector<float>* p_sigma, 
    const std::vector<float>* p_density, 
    float spacing_z, 
    float* sigma_max, 
    float energy, 
    float sourcesize
)
{
    std::vector<float> tmp_rglength (p_sigma->size(), 0);
    int first_non_null_loc = 0;
    spacing_z = spacing_z/10; // converted to cm (the Highland formula is in cm!)

    /* initializiation of all the french_fries, except sigma, which is the output and calculate later */
    for(int i = 0; i < (int) p_sigma->size();i++)
    {
        if(i == 0)
        {
            tmp_rglength[i] = (*p_sigma)[i]; // remember that at this point french_fries_sigma is only a rglngth function without compensator
            (*p_sigma)[i] = 0;
        } 
        else 
        {
            tmp_rglength[i] = (*p_sigma)[i]-(*p_sigma)[i-1]; // rglength in the pixel
            (*p_sigma)[i] = 0;
        }
    }

    //We can now compute the sigma french_fries!!!!

    /* Step 1: the sigma is filled with zeros, so we let them at 0 as long as rg_length is 0, meaning the ray is out of the volume */
    /* we mark the first pixel in the volume, and the calculations will start with this one */
    for (int i = 0; i < (int) p_sigma->size(); i++)
    {
        if (tmp_rglength[i] > 0)
        {
            first_non_null_loc = i;
            break;
        }
        if (i == p_sigma->size())
        {
            first_non_null_loc = p_sigma->size()-1;
            printf("\n the french_fries is completely zeroed, the ray seems to not intersect the volume\n");
            return;
        }
    }

    /* Step 2: Each pixel in the volume will receive its sigma (in reality y0) value, according to the differential Highland formula */

    float energy_callback = energy;
    float mc2 = 939.4f;          /* proton mass at rest (MeV) */
    float c = 299792458.0f;        /* speed of light (m/s2) */

    float p = 0.0;              /* Proton momentum (passed in)          */
    float v = 0.0;              /* Proton velocity (passed in)          */
    float stop = 0;		/* stopping power energy (MeV.cm2/g) */
	
    float sum = 0.0;		/* integration expressions, right part of equation */
    float function_to_be_integrated; /* right term to be integrated in the Highland equation */
    float inverse_rad_length_integrated = 0; /* and left part */
    float y0 = 0;               /* y0 value used to update the french_fries values */
    float inv_rad_length;       /* 1/rad_length - used in the integration */
    float step;                 /* step of integration, will depends on the radiologic length */

    float POI_depth;            /* depth of the point of interest (where is calculated the sigma value)in cm - centered at the pixel center*/
    float pixel_depth;          /* depth of the contributing pixel to total sigma (in cm) - center between 2 pixels, the difference in rglength comes from the center of the previous pixel to the center of this pixel*/
    double sigma_source;
    double sigma_range_compensator;
    double sigma_patient;

    for (int i = first_non_null_loc; i < (int) p_sigma->size(); i++)
    {
        energy = energy_callback; // we reset the parameters for each sigma calculation
        sum = 0;
        inverse_rad_length_integrated = 0;

        POI_depth = (float) (i+0.5)*spacing_z;

        /*integration */
        for (int j = first_non_null_loc; j <= i && energy > 0;j++)
        {

            /* p & v are updated */

            p= sqrt(2*energy*mc2+energy*energy)/c; // in MeV.s.m-1
            v= c*sqrt(1-pow((mc2/(energy+mc2)),2)); //in m.s-1

            if (i == j)
            {
                pixel_depth = (j+.25f)*spacing_z; // we integrate only up to the voxel center, not the whole pixel
                step = spacing_z/2;
            }
            else
            {
                pixel_depth = (j+0.5f)*spacing_z;
                step = spacing_z;
            }
            
            inv_rad_length = 1.0f / LR_interpolation((*p_density)[j]);

            function_to_be_integrated = (pow(((POI_depth - pixel_depth)/(p*v)),2) * inv_rad_length); //i in cm
            sum += function_to_be_integrated*step;

            inverse_rad_length_integrated += step * inv_rad_length;

            /* energy is updated after passing through dz */
            
            stop = (float) getstop(energy)* WER_interpolation((*p_density)[j]) * (*p_density)[j]; // dE/dx_mat = dE /dx_watter * WER * density (lut in g/cm2)
            energy = energy - stop*step;
        }

        if (energy  <= 0) // sigma formula is not defined anymore
        {
            return; // we can exit as the rest of the french_fries_sigma equals already 0
        }
        else // that means we reach the POI pixel and we can store the y0 value
        {
            sigma_source = sourcesize * ((20 + POI_depth) / 187);
            sigma_range_compensator = 0.331 * 0.00313 * (20 + POI_depth + 4.4); // 4.4 is the fraction of the RC - effective scattering depth on 13cm
            sigma_patient = 14.10f *(1.0f+1.0f/9.0f*log10(inverse_rad_length_integrated))* (float) sqrt(sum); // in cm
            (*p_sigma)[i] = 10* ( //*10 because sigma is used later in mm
                sqrt(sigma_source * sigma_source + sigma_range_compensator * sigma_range_compensator + sigma_patient * sigma_patient));

            if (*sigma_max < (*p_sigma)[i])
            {
                *sigma_max = (*p_sigma) [i];
            }
        }
    }
}

void length_to_sigma_photon(std::vector<float>* p_sigma, std::vector<float>* p_density, float spacing_z, float* sigma_max, float energy, float sourcesize)
{
    /* initializiation of all the french_fries, except sigma, which is the output and calculate later */
    for(int i = 0; i < (int) p_sigma->size();i++)
    {
        if ((*p_sigma)[i] <= 0) {(*p_sigma)[i] = 0.1;} // definition of a rough sigma for protons, linear until 1.0 cm (0.7mm) then still 0.7
        else if ((*p_sigma)[i] <= 10) { (*p_sigma)[i] = (*p_sigma)[i] /10 * 0.7; }
        else { (*p_sigma)[i] = 0.7;}

        if (*sigma_max < (*p_sigma)[i])
        {
            *sigma_max = (*p_sigma) [i];
        }
    }
    return;
}

void
compute_dose_ray_desplanques(Volume* dose_volume, Volume::Pointer ct_vol, Rpl_volume* rpl_volume, Rpl_volume* sigma_volume, Rpl_volume* ct_rpl_volume, Ion_beam* beam, Volume::Pointer final_dose_volume, const Ion_pristine_peak* ppp, float normalization_dose)
{
    if (ppp->weight <= 0)
    {
        return;
    }
    int ijk_idx[3] = {0,0,0};
    int ijk_travel[3] = {0,0,0};
    double xyz_travel[3] = {0.0,0.0,0.0};

    int ap_ij[2] = {1,0};
    int dim[2] = {0,0};

    double ray_bev[3] = {0,0,0};

    double xyz_ray_center[3] = {0.0, 0.0, 0.0};
    double xyz_ray_pixel_center[3] = {0.0, 0.0, 0.0};

    double entrance_bev[3] = {0.0f, 0.0f, 0.0f}; // coordinates of intersection with the volume in the bev frame
    double entrance_length = 0;
    double aperture_bev[3] = {0.0f, 0.0f, 0.0f}; // coordinates of intersection with the aperture in the bev frame
    double distance = 0; // distance from the aperture to the POI
    double tmp[3] = {0.0f, 0.0f, 0.0f};

    double PB_density = 1/(rpl_volume->get_aperture()->get_spacing(0) * rpl_volume->get_aperture()->get_spacing(1));

    double dose_norm = 0; // factor that normalize dose to be = 1 at max.
    dose_norm = get_dose_norm('f', ppp->E0, PB_density); //the Hong algorithm has no PB density, everything depends on the number of sectors


    double ct_density = 0;
    double sigma = 0;
    int sigma_x3 = 0;
    double rg_length = 0;
    double radius = 0;

    float central_axis_dose = 0;
    double r_over_sigma = 0;
    int r_over_sigma_round = 0;
    float off_axis_factor = 0;

    int idx = 0; // index to travel in the dose volume
    int idx_bev = 0; // second index for reconstructing the final image
    bool test = true;
    bool* in = &test;

    double vec_antibug_prt[3] = {0.0,0.0,0.0};

    int i_min = 0;
    int i_max = 0;
    int j_min = 0;
    int j_max = 0;

    dim[0] = sigma_volume->get_aperture()->get_dim(0);
    dim[1] = sigma_volume->get_aperture()->get_dim(1);

    float* img = (float*) dose_volume->img;
    float* ct_img = (float*) ct_vol->img;
    float* rpl_image = (float*) rpl_volume->get_vol()->img;

    double dist = 0;
    int offset_step = 0;

    for (int i = 0; i < dim[0]*dim[1];i++)
    {
        Ray_data* ray_data = &rpl_volume->get_Ray_data()[i];

        ap_ij[1] = i / dim[0];
        ap_ij[0] = i- ap_ij[1]*dim[0];

        vec3_cross(vec_antibug_prt, rpl_volume->get_aperture()->pdn, rpl_volume->get_proj_volume()->get_nrm());

        ray_bev[0] = vec3_dot(ray_data->ray, vec_antibug_prt);
        ray_bev[1] = vec3_dot(ray_data->ray, rpl_volume->get_aperture()->pdn);
        ray_bev[2] = -vec3_dot(ray_data->ray, rpl_volume->get_proj_volume()->get_nrm()); // ray_beam_eye_view is already normalized

        /* Calculation of the coordinates of the intersection of the ray with the clipping plane */
        entrance_length = vec3_dist(rpl_volume->get_proj_volume()->get_src(), ray_data->cp);
        entrance_length += (double) ray_data->step_offset * rpl_volume->get_proj_volume()->get_step_length ();

        vec3_copy(entrance_bev, ray_bev);
        vec3_scale2(entrance_bev, entrance_length);

        if (ray_bev[2]  > DRR_BOUNDARY_TOLERANCE)
        {
            for(int k = 0; k < dose_volume->dim[2];k++)
            {
                find_xyz_center(xyz_ray_center, ray_bev, vec3_len(entrance_bev),k);
                distance = vec3_dist(entrance_bev, xyz_ray_center);

                ct_density = ct_rpl_volume->get_rgdepth(ap_ij, distance);
				
                if (ct_density <= 0) // no medium, no dose... (air = void)
                {
                    continue;
                }
                else
                {
                    rg_length = rpl_volume->get_rgdepth(ap_ij, distance);
                    central_axis_dose = ppp->lookup_energy((float)rg_length);

                    sigma = sigma_volume->get_rgdepth(ap_ij, distance);
                    sigma_x3 = (int) ceil(3 * sigma);
                    rg_length = rpl_volume->get_rgdepth(ap_ij, distance);

                    /* We defined the grid to be updated, the pixels that receive dose from the ray */
                    /* We don't check to know if we are still in the matrix because the matrix was build to contain all pixels with a 3 sigma_max margin */
                    find_ijk_pixel(ijk_idx, xyz_ray_center, dose_volume);
                    
                    i_min = ijk_idx[0] - sigma_x3;
                    i_max = ijk_idx[0] + sigma_x3;
                    j_min = ijk_idx[1] - sigma_x3;
                    j_max = ijk_idx[1] + sigma_x3;

                    central_axis_dose = beam->lookup_sobp_dose((float) rg_length);
                    
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

                            find_xyz_from_ijk(xyz_travel,dose_volume,ijk_travel);
                            
                            radius = vec3_dist(xyz_travel,xyz_ray_center);                            
                            if (sigma == 0)
                            {
                                off_axis_factor = 1;
                            }
                            else if (radius / sigma >=3)
                            {
                                off_axis_factor = 0;
                            }
                            else
                            {
                                off_axis_factor = double_gaussian_interpolation(xyz_ray_center, xyz_travel,sigma, (double*) dose_volume->spacing);
                            }
                            img[idx] += normalization_dose * beam->get_beamWeight() * central_axis_dose * off_axis_factor * (float) ppp->weight / dose_norm; // SOBP is weighted by the weight of the pristine peak
                        }
                    }
                }
            }
        }
        else
        {
            printf("Ray[%d] is not directed forward: z,x,y (%lg, %lg, %lg) \n", i, ray_data->ray[0], ray_data->ray[1], ray_data->ray[2]);
        }
    }

    float* final_dose_img = (float*) final_dose_volume->img;

    int ijk[3] = {0,0,0};
    float ijk_bev[3] = {0,0,0};
    int ijk_bev_trunk[3];
    double xyz_room[3] = {0.0,0.0,0.0};
    float xyz_bev[3] = {0.0,0.0,0.0};

    plm_long mijk_f[3];
    plm_long mijk_r[3];

    float li_frac1[3];
    float li_frac2[3];

    int ct_dim[3] = {ct_vol->dim[0], ct_vol->dim[1], ct_vol->dim[2]};

    for (ijk[0] = 0; ijk[0] < ct_dim[0]; ijk[0]++)
    {
        for (ijk[1] = 0; ijk[1] < ct_dim[1]; ijk[1]++)
        {
            for (ijk[2] = 0; ijk[2] < ct_dim[2]; ijk[2]++)
            {
                idx = ijk[0] + ct_dim[0] *(ijk[1] + ijk[2] * ct_dim[1]);
                if ( ct_img[idx] > -1000) // in air we have no dose, we let the voxel number at 0!
                {   
                    final_dose_volume->get_xyz_from_ijk(xyz_room, ijk);

                    /* xyz contains the coordinates of the pixel in the room coordinates */
                    /* we now calculate the coordinates of this pixel in the dose_volume coordinates */
                    /* need to be fixed after the extrinsic homogeneous coordinates is fixed */

                    vec3_sub3(tmp, rpl_volume->get_proj_volume()->get_src(), xyz_room);
                   
                    xyz_bev[0] = (float) -vec3_dot(tmp, vec_antibug_prt);
                    xyz_bev[1] = (float) -vec3_dot(tmp, rpl_volume->get_aperture()->pdn);
                    xyz_bev[2] = (float) vec3_dot(tmp, rpl_volume->get_proj_volume()->get_nrm());

                    dose_volume->get_ijk_from_xyz(ijk_bev,xyz_bev, in);
                    if (*in == true)
                    {
                        dose_volume->get_ijk_from_xyz(ijk_bev_trunk, xyz_bev, in);

                        idx_bev = ijk_bev_trunk[0] + ijk_bev[1]*dose_volume->dim[0] + ijk_bev[2] * dose_volume->dim[0] * dose_volume->dim[1];
                        li_clamp_3d(ijk_bev, mijk_f, mijk_r, li_frac1, li_frac2, dose_volume);

                        final_dose_img[idx] += li_value(li_frac1[0], li_frac2[0], li_frac1[1], li_frac2[1], li_frac1[2], li_frac2[2], idx_bev, img, dose_volume);
                    }
                    else
                    {
                        final_dose_img[idx] += 0;
                    }

                }
            }
            
        }
        
    }
    return;
}

void
compute_dose_ray_desplanques(Volume* dose_volume, Volume::Pointer ct_vol, Rpl_volume* rpl_volume, Rpl_volume* sigma_volume, Rpl_volume* ct_rpl_volume, Photon_beam* beam, Volume::Pointer final_dose_volume, const Photon_depth_dose* ppp, float normalization_dose)
{
    int ijk_idx[3] = {0,0,0};
    int ijk_travel[3] = {0,0,0};
    double xyz_travel[3] = {0.0,0.0,0.0};

    int ap_ij[2] = {1,0};
    int dim[2] = {0,0};

    double ray_bev[3] = {0,0,0};

    double xyz_ray_center[3] = {0.0, 0.0, 0.0};
    double xyz_ray_pixel_center[3] = {0.0, 0.0, 0.0};

    double entrance_bev[3] = {0.0f, 0.0f, 0.0f}; // coordinates of intersection with the volume in the bev frame
    double entrance_length = 0;
    double aperture_bev[3] = {0.0f, 0.0f, 0.0f}; // coordinates of intersection with the aperture in the bev frame
    double distance = 0; // distance from the aperture to the POI
    double tmp[3] = {0.0f, 0.0f, 0.0f};

    double ct_density = 0;
    double sigma = 0;
    int sigma_x3 = 0;
    double rg_length = 0;
    double radius = 0;

    float central_axis_dose = 0;
    double r_over_sigma = 0;
    int r_over_sigma_round = 0;
    float off_axis_factor = 0;

    int idx = 0; // index to travel in the dose volume
    int idx_bev = 0; // second index for reconstructing the final image
    bool test = true;
    bool* in = &test;

    double vec_antibug_prt[3] = {0.0,0.0,0.0};

    int i_min = 0;
    int i_max = 0;
    int j_min = 0;
    int j_max = 0;

    dim[0] = sigma_volume->get_aperture()->get_dim(0);
    dim[1] = sigma_volume->get_aperture()->get_dim(1);

    float* img = (float*) dose_volume->img;
    float* ct_img = (float*) ct_vol->img;
    float* rpl_image = (float*) rpl_volume->get_vol()->img;

    double dist = 0;
    int offset_step = 0;

    for (int i = 0; i < dim[0]*dim[1];i++)
    {
        Ray_data* ray_data = &rpl_volume->get_Ray_data()[i];

        ap_ij[1] = i / dim[0];
        ap_ij[0] = i- ap_ij[1]*dim[0];

        vec3_cross(vec_antibug_prt, rpl_volume->get_aperture()->pdn, rpl_volume->get_proj_volume()->get_nrm());

        ray_bev[0] = vec3_dot(ray_data->ray, vec_antibug_prt);
        ray_bev[1] = vec3_dot(ray_data->ray, rpl_volume->get_aperture()->pdn);
        ray_bev[2] = -vec3_dot(ray_data->ray, rpl_volume->get_proj_volume()->get_nrm()); // ray_beam_eye_view is already normalized

        /* Calculation of the coordinates of the intersection of the ray with the clipping plane */
        entrance_length = vec3_dist(rpl_volume->get_proj_volume()->get_src(), ray_data->cp);
        entrance_length += (double) ray_data->step_offset * rpl_volume->get_proj_volume()->get_step_length ();

        vec3_copy(entrance_bev, ray_bev);
        vec3_scale2(entrance_bev, entrance_length);

        if (ray_bev[2]  > DRR_BOUNDARY_TOLERANCE)
        {
            for(int k = 0; k < dose_volume->dim[2];k++)
            {
                find_xyz_center(xyz_ray_center, ray_bev, vec3_len(entrance_bev),k);
                distance = vec3_dist(entrance_bev, xyz_ray_center);

                ct_density = ct_rpl_volume->get_rgdepth(ap_ij, distance);
				
                if (ct_density <= 0) // no medium, no dose... (air = void)
                {
                    continue;
                }
                else
                {
                    rg_length = rpl_volume->get_rgdepth(ap_ij, distance);
                    central_axis_dose = ppp->lookup_energy((float)rg_length);

                    sigma = sigma_volume->get_rgdepth(ap_ij, distance);
                    sigma_x3 = (int) ceil(3 * sigma);
                    rg_length = rpl_volume->get_rgdepth(ap_ij, distance);

                    /* We defined the grid to be updated, the pixels that receive dose from the ray */
                    /* We don't check to know if we are still in the matrix because the matrix was build to contain all pixels with a 3 sigma_max margin */
                    find_ijk_pixel(ijk_idx, xyz_ray_center, dose_volume);
                    
                    i_min = ijk_idx[0] - sigma_x3;
                    i_max = ijk_idx[0] + sigma_x3;
                    j_min = ijk_idx[1] - sigma_x3;
                    j_max = ijk_idx[1] + sigma_x3;

                    central_axis_dose = beam->lookup_sobp_dose((float) rg_length);
                    
                    for (int i2 = i_min; i2 <= i_max; i2++)
                    {
                        for (int j2 = j_min; j2 <= j_max; j2++)
                        {
                            idx = i2 + (dose_volume->dim[0] * (j2 + dose_volume->dim[1] * k));

                            ijk_travel[0] = i2;
                            ijk_travel[1] = j2;
                            ijk_travel[2] = k;

                            find_xyz_from_ijk(xyz_travel,dose_volume,ijk_travel);
                            
                            radius = vec3_dist(xyz_travel,xyz_ray_center);                            
                            if (sigma == 0)
                            {
                                off_axis_factor = 1;
                            }
                            else if (radius / sigma >=3)
                            {
                                off_axis_factor = 0;
                            }
                            else
                            {
                                off_axis_factor = double_gaussian_interpolation(xyz_ray_center, xyz_travel,sigma, (double*) dose_volume->spacing);
                            }
                            img[idx] += normalization_dose * central_axis_dose * off_axis_factor * (float) ppp->weight; // SOBP is weighted by the weight of the pristine peak
                        }
                    }
                }
            }
        }
        else
        {
            printf("Ray[%d] is not directed forward: z,x,y (%lg, %lg, %lg) \n", i, ray_data->ray[0], ray_data->ray[1], ray_data->ray[2]);
        }
    }

    float* final_dose_img = (float*) final_dose_volume->img;

    int ijk[3] = {0,0,0};
    float ijk_bev[3] = {0,0,0};
    int ijk_bev_trunk[3];
    double xyz_room[3] = {0.0,0.0,0.0};
    float xyz_bev[3] = {0.0,0.0,0.0};

    plm_long mijk_f[3];
    plm_long mijk_r[3];

    float li_frac1[3];
    float li_frac2[3];

    int ct_dim[3] = {ct_vol->dim[0], ct_vol->dim[1], ct_vol->dim[2]};

    for (ijk[0] = 0; ijk[0] < ct_dim[0]; ijk[0]++)
    {
        for (ijk[1] = 0; ijk[1] < ct_dim[1]; ijk[1]++)
        {
            for (ijk[2] = 0; ijk[2] < ct_dim[2]; ijk[2]++)
            {
                idx = ijk[0] + ct_dim[0] *(ijk[1] + ijk[2] * ct_dim[1]);
                if ( ct_img[idx] > -1000) // in air we have no dose, we let the voxel number at 0!
                {   
                    final_dose_volume->get_xyz_from_ijk(xyz_room, ijk);

                    /* xyz contains the coordinates of the pixel in the room coordinates */
                    /* we now calculate the coordinates of this pixel in the dose_volume coordinates */
                    /* need to be fixed after the extrinsic homogeneous coordinates is fixed */

                    vec3_sub3(tmp, rpl_volume->get_proj_volume()->get_src(), xyz_room);
                   
                    xyz_bev[0] = (float) -vec3_dot(tmp, vec_antibug_prt);
                    xyz_bev[1] = (float) -vec3_dot(tmp, rpl_volume->get_aperture()->pdn);
                    xyz_bev[2] = (float) vec3_dot(tmp, rpl_volume->get_proj_volume()->get_nrm());

                    dose_volume->get_ijk_from_xyz(ijk_bev,xyz_bev, in);
                    if (*in == true)
                    {
                        dose_volume->get_ijk_from_xyz(ijk_bev_trunk, xyz_bev, in);

                        idx_bev = ijk_bev_trunk[0] + ijk_bev[1]*dose_volume->dim[0] + ijk_bev[2] * dose_volume->dim[0] * dose_volume->dim[1];
                        li_clamp_3d(ijk_bev, mijk_f, mijk_r, li_frac1, li_frac2, dose_volume);

                        final_dose_img[idx] += li_value(li_frac1[0], li_frac2[0], li_frac1[1], li_frac2[1], li_frac1[2], li_frac2[2], idx_bev, img, dose_volume);
                    }
                    else
                    {
                        final_dose_img[idx] += 0;
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
    const Rpl_volume* rpl_volume, 
    const Rpl_volume* sigma_volume, 
    const Rpl_volume* ct_rpl_volume, 
    const Ion_beam* beam, 
    Rpl_volume* rpl_dose_volume, 
    const Aperture::Pointer ap, 
    const Ion_pristine_peak* ppp, 
    const int* margins, 
    float normalization_dose
)
{
    int ap_ij_lg[2] = {0,0};
    int ap_ij_sm[2] = {0,0};
    int dim_lg[3] = {0,0,0};
    int dim_sm[3] = {0,0,0};

    double ct_density = 0;
    double sigma = 0;
    double sigma_x3 = 0;
    double rg_length = 0;

    double central_ray_xyz[3] = {0,0,0};
    double travel_ray_xyz[3] = {0,0,0};

    float central_axis_dose = 0;
    double r_over_sigma = 0;
    int r_over_sigma_round = 0;
    float off_axis_factor = 0;

    double PB_density = 1 / (rpl_volume->get_aperture()->get_spacing(0) * rpl_volume->get_aperture()->get_spacing(1));

    double dose_norm = get_dose_norm ('g', ppp->E0, PB_density);
    //the Hong algorithm has no PB density, everything depends on the number of sectors

    int idx2d_sm = 0;
    int idx2d_lg = 0;
    int idx3d_sm = 0;
    int idx3d_lg = 0;
    int idx3d_travel = 0;

    double minimal_lateral = 0;
    double lateral_step[2] = {0,0};
    int i_min = 0;
    int i_max = 0;
    int j_min = 0;
    int j_max = 0;

    dim_lg[0] = rpl_dose_volume->get_vol()->dim[0];
    dim_lg[1] = rpl_dose_volume->get_vol()->dim[1];
    dim_lg[2] = rpl_dose_volume->get_vol()->dim[2];

    dim_sm[0] = rpl_volume->get_vol()->dim[0];
    dim_sm[1] = rpl_volume->get_vol()->dim[1];
    dim_sm[2] = rpl_volume->get_vol()->dim[2];

    float* rpl_img = (float*) rpl_volume->get_vol()->img;
    float* sigma_img = (float*) sigma_volume->get_vol()->img;
    float* rpl_dose_img = (float*) rpl_dose_volume->get_vol()->img;

    double dist = 0;
    double radius = 0;

    /* Creation of the rpl_volume containing the coordinates xyz (beam eye view) and the CT density vol*/
    std::vector<double> xyz_init (4,0);
    std::vector< std::vector<double> > xyz_coor_vol (dim_lg[0]*dim_lg[1]*dim_lg[2], xyz_init);
    std::vector<double> CT_density_vol (dim_lg[0]*dim_lg[1]*dim_lg[2], 0);

    calculate_rpl_coordinates_xyz (&xyz_coor_vol, rpl_dose_volume);

    for (int m = 0; m < dim_lg[0] * dim_lg[1] * dim_lg[2]; m++)
    {
        rpl_dose_img[m] = 0;
    }

    /* calculation of the lateral steps in which the dose is searched constant with depth */
    std::vector <double> lateral_minimal_step (dim_lg[2],0);
    std::vector <double> lateral_step_x (dim_lg[2],0);
    std::vector <double> lateral_step_y (dim_lg[2],0);

    minimal_lateral = ap->get_spacing(0);
    
    if (minimal_lateral < ap->get_spacing(1))
    {
        minimal_lateral = ap->get_spacing(1);
    }
    for (int k = 0; k < dim_lg[2]; k++)
    {
        lateral_minimal_step[k] = (rpl_volume->get_front_clipping_plane() + (double) k) * minimal_lateral / rpl_volume->get_aperture()->get_distance();
        lateral_step_x[k] = (rpl_volume->get_front_clipping_plane() + (double) k) * ap->get_spacing(0) / rpl_volume->get_aperture()->get_distance();
        lateral_step_y[k] = (rpl_volume->get_front_clipping_plane() + (double) k) * ap->get_spacing(1) / rpl_volume->get_aperture()->get_distance();
    }

    /* calculation of the dose in the rpl_volume */
    for (ap_ij_lg[0] = margins[0]; ap_ij_lg[0] < rpl_dose_volume->get_vol()->dim[0]-margins[0]; ap_ij_lg[0]++){
        for (ap_ij_lg[1] = margins[1]; ap_ij_lg[1] < rpl_dose_volume->get_vol()->dim[1]-margins[1]; ap_ij_lg[1]++){

            ap_ij_sm[0] = ap_ij_lg[0] - margins[0];
            ap_ij_sm[1] = ap_ij_lg[1] - margins[1];

            idx2d_lg = ap_ij_lg[1] * dim_lg[0] + ap_ij_lg[0];
            idx2d_sm = ap_ij_sm[1] * dim_sm[0] + ap_ij_sm[0];

            Ray_data* ray_data = &rpl_dose_volume->get_Ray_data()[idx2d_lg];

            /* When would this happen? */
            if (-vec3_dot(ray_data->ray, rpl_dose_volume->get_proj_volume()->get_nrm()) <= DRR_BOUNDARY_TOLERANCE)
            {
                continue;
            }

            for (int k = 0; k < dim_lg[2]; k++)
            {
                idx3d_lg = idx2d_lg + k * dim_lg[0]*dim_lg[1];
                idx3d_sm = idx2d_sm + k * dim_sm[0]*dim_sm[1];

                central_ray_xyz[0] = xyz_coor_vol[idx3d_lg][0];
                central_ray_xyz[1] = xyz_coor_vol[idx3d_lg][1];
                central_ray_xyz[2] = xyz_coor_vol[idx3d_lg][2];

                lateral_step[0] = lateral_step_x[k];
                lateral_step[1] = lateral_step_x[k];

                ct_density = (double) rpl_img[idx3d_sm];
                if (ct_density <= 0) // no medium, no dose... (air = void) or we are not in the aperture but in the margins fr the penubras
                {
                    continue;
                }


                rg_length = rpl_img[idx3d_sm];
                central_axis_dose = ppp->lookup_energy(rg_length);

                if (central_axis_dose <= 0) 
                {
                    continue;
                } // no dose on the axis, no dose scattered

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

                for (int i1 = i_min; i1 <= i_max; i1++) {
                    for (int j1 = j_min; j1 <= j_max; j1++) {

                        idx3d_travel = k * dim_lg[0]*dim_lg[1] + j1 * dim_lg[0] + i1;

                        travel_ray_xyz[0] = xyz_coor_vol[idx3d_travel][0];
                        travel_ray_xyz[1] = xyz_coor_vol[idx3d_travel][1];
                        travel_ray_xyz[2] = xyz_coor_vol[idx3d_travel][2];
								
                        radius = vec3_dist(travel_ray_xyz, central_ray_xyz);                            
                        if (sigma == 0)
                        {
                            off_axis_factor = 1;
                        }
                        else if (radius / sigma >=3)
                        {
                            off_axis_factor = 0;
                        }
                        else
                        {
                            off_axis_factor = double_gaussian_interpolation(central_ray_xyz, travel_ray_xyz, sigma, lateral_step);
                        }

                        rpl_dose_img[idx3d_travel] += normalization_dose * beam->get_beamWeight() * central_axis_dose * off_axis_factor * (float) ppp->weight  / dose_norm; // SOBP is weighted by the weight of the pristine peak
                    } //for j1
                } //for i1
            } // for k
        } // ap_ij[1]
    } // ap_ij[0]
}

void 
compute_dose_ray_sharp(Volume::Pointer ct_vol, Rpl_volume* rpl_volume, Rpl_volume* sigma_volume, Rpl_volume* ct_rpl_volume, Photon_beam* beam, Rpl_volume* rpl_dose_volume, Aperture::Pointer ap, const Photon_depth_dose* ppp, int* margins, float normalization_dose)
{
    int ap_ij_lg[2] = {0,0};
    int ap_ij_sm[2] = {0,0};
    int dim_lg[3] = {0,0,0};
    int dim_sm[3] = {0,0,0};

    double ct_density = 0;
    double sigma = 0;
    double sigma_x3 = 0;
    double rg_length = 0;

    double central_ray_xyz[3] = {0,0,0};
    double travel_ray_xyz[3] = {0,0,0};

    float central_axis_dose = 0;
    double r_over_sigma = 0;
    int r_over_sigma_round = 0;
    float off_axis_factor = 0;

    int idx2d_sm = 0;
    int idx2d_lg = 0;
    int idx3d_sm = 0;
    int idx3d_lg = 0;
    int idx3d_travel = 0;

    double minimal_lateral = 0;
    double lateral_step[2] = {0,0};
    int i_min = 0;
    int i_max = 0;
    int j_min = 0;
    int j_max = 0;

    dim_lg[0] = rpl_dose_volume->get_vol()->dim[0];
    dim_lg[1] = rpl_dose_volume->get_vol()->dim[1];
    dim_lg[2] = rpl_dose_volume->get_vol()->dim[2];

    dim_sm[0] = rpl_volume->get_vol()->dim[0];
    dim_sm[1] = rpl_volume->get_vol()->dim[1];
    dim_sm[2] = rpl_volume->get_vol()->dim[2];

    float* rpl_img = (float*) rpl_volume->get_vol()->img;
    float* sigma_img = (float*) sigma_volume->get_vol()->img;
    float* rpl_dose_img = (float*) rpl_dose_volume->get_vol()->img;
    float* ct_rpl_img = (float*) ct_rpl_volume->get_vol()->img;

    double dist = 0;
    double radius = 0;
    
    /* Creation of the rpl_volume containing the coordinates xyz (beam eye view) and the CT density vol*/
    std::vector<double> xyz_init (4,0);
    std::vector< std::vector<double> > xyz_coor_vol (dim_lg[0]*dim_lg[1]*dim_lg[2], xyz_init);
    std::vector<double> CT_density_vol (dim_lg[0]*dim_lg[1]*dim_lg[2], 0);

    calculate_rpl_coordinates_xyz(&xyz_coor_vol, rpl_dose_volume);
    copy_rpl_density(&CT_density_vol, rpl_dose_volume);

    for (int m = 0; m < dim_lg[0] * dim_lg[1] * dim_lg[2]; m++)
    {
        rpl_dose_img[m] = 0;
    }

    /* calculation of the lateral steps in which the dose is searched constant with depth */
    std::vector <double> lateral_minimal_step (dim_lg[2],0);
    std::vector <double> lateral_step_x (dim_lg[2],0);
    std::vector <double> lateral_step_y (dim_lg[2],0);

    minimal_lateral = ap->get_spacing(0);
    
    if (minimal_lateral < ap->get_spacing(1))
    {
        minimal_lateral = ap->get_spacing(1);
    }
    for(int k = 0; k < dim_lg[2]; k++)
    {
        lateral_minimal_step[k] = (rpl_volume->get_front_clipping_plane() + (double) k) * minimal_lateral / rpl_volume->get_aperture()->get_distance();
        lateral_step_x[k] = (rpl_volume->get_front_clipping_plane() + (double) k) * ap->get_spacing(0) / rpl_volume->get_aperture()->get_distance();
        lateral_step_y[k] = (rpl_volume->get_front_clipping_plane() + (double) k) * ap->get_spacing(1) / rpl_volume->get_aperture()->get_distance();
    }

    /* calculation of the dose in the rpl_volume */
    for (ap_ij_lg[0] = margins[0]; ap_ij_lg[0] < rpl_dose_volume->get_vol()->dim[0]-margins[0]; ap_ij_lg[0]++){
        printf("%d_",ap_ij_lg[0]);

        for (ap_ij_lg[1] = margins[1]; ap_ij_lg[1] < rpl_dose_volume->get_vol()->dim[1]-margins[1]; ap_ij_lg[1]++){
            ap_ij_sm[0] = ap_ij_lg[0] - margins[0];
            ap_ij_sm[1] = ap_ij_lg[1] - margins[1];

            idx2d_lg = ap_ij_lg[1] * dim_lg[0] + ap_ij_lg[0];
            idx2d_sm = ap_ij_sm[1] * dim_sm[0] + ap_ij_sm[0];

            Ray_data* ray_data = &rpl_dose_volume->get_Ray_data()[idx2d_lg];

            if (-vec3_dot(ray_data->ray, rpl_dose_volume->get_proj_volume()->get_nrm()) > DRR_BOUNDARY_TOLERANCE)
            {
                for(int k = 0; k < dim_lg[2]; k++)
                {
                    idx3d_lg = idx2d_lg + k * dim_lg[0]*dim_lg[1];
                    idx3d_sm = idx2d_sm + k * dim_sm[0]*dim_sm[1];

                    central_ray_xyz[0] = xyz_coor_vol[idx3d_lg][0];
                    central_ray_xyz[1] = xyz_coor_vol[idx3d_lg][1];
                    central_ray_xyz[2] = xyz_coor_vol[idx3d_lg][2];

                    lateral_step[0] = lateral_step_x[k];
                    lateral_step[1] = lateral_step_x[k];

                    ct_density = (double) rpl_img[idx3d_sm];
                    if (ct_density <= 0) // no medium, no dose... (air = void) or we are not in the aperture but in the margins fr the penubras
                    {
                        continue;
                    }
                    else
                    {
                        rg_length = rpl_img[idx3d_sm];
                        central_axis_dose = ppp->lookup_energy(rg_length);

                        if (central_axis_dose <=0) 
                        {
                            continue;
                        } // no dose on the axis, no dose scattered

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
    
                        for (int i1 = i_min; i1 <= i_max; i1++){
                            for (int j1 = j_min; j1 <= j_max; j1++){
                                
                                idx3d_travel = k * dim_lg[0]*dim_lg[1] + j1 * dim_lg[0] + i1;

                                travel_ray_xyz[0] = xyz_coor_vol[idx3d_travel][0];
                                travel_ray_xyz[1] = xyz_coor_vol[idx3d_travel][1];
                                travel_ray_xyz[2] = xyz_coor_vol[idx3d_travel][2];
								
                                radius = vec3_dist(travel_ray_xyz, central_ray_xyz);                            
                                if (sigma == 0)
                                {
                                    off_axis_factor = 1;
                                }
                                else if (radius / sigma >=3)
                                {
                                    off_axis_factor = 0;
                                }
                                else
                                {
                                    off_axis_factor = double_gaussian_interpolation(central_ray_xyz, travel_ray_xyz, sigma, lateral_step);
                                }

                                rpl_dose_img[idx3d_travel] += normalization_dose * central_axis_dose * off_axis_factor * (float) ppp->weight; // SOBP is weighted by the weight of the pristine peak
                            } //for j1
                        } //for i1
                    } // else
                } // for k
            } // if
        } // ap_ij[1]
    } // ap_ij[0]
    return;
}

void compute_dose_ray_shackleford(Volume::Pointer dose_vol, Ion_plan* plan, const Ion_pristine_peak* ppp, std::vector<double>* area, std::vector<double>* xy_grid, int radius_sample, int theta_sample)
{
    int ijk[3] = {0,0,0};
    double xyz[4] = {0,0,0,1};
    double xyz_travel[4] = {0,0,0,1};
    double tmp_xy[4] = {0,0,0,1};
    double tmp_cst = 0;

    double dose_norm = get_dose_norm('h', ppp->E0, 1); //the Hong algorithm has no PB density, everything depends on the number of sectors


    int idx = 0;
	
    int ct_dim[3] = {dose_vol->dim[0], dose_vol->dim[1], dose_vol->dim[2]};
    double vec_ud[4] = {0,0,0,1};
    double vec_rl[4] = {0,0,0,1};

    float* ct_img = (float*) plan->get_patient_volume()->img;
    float* dose_img = (float*) dose_vol->img;

    double sigma_travel = 0;
    double sigma_3 = 0;
    double rg_length = 0;
    double central_sector_dose = 0;
    double radius = 0;
    double theta = 0;
    double r_s = 0;

    vec3_copy(vec_ud, plan->get_aperture()->pdn);
    vec3_normalize1(vec_ud);
    vec3_copy(vec_rl, plan->get_aperture()->prt);
    vec3_normalize1(vec_rl);

    for (ijk[0] = 0; ijk[0] < ct_dim[0]; ijk[0]++){
        for (ijk[1] = 0; ijk[1] < ct_dim[1]; ijk[1]++){
            for (ijk[2] = 0; ijk[2] < ct_dim[2]; ijk[2]++){
                idx = ijk[0] + ct_dim[0] * (ijk[1] + ct_dim[1] * ijk[2]);

                if (ct_img[idx] <= -1000) {continue;} // if this pixel is in the air, no dose delivered

                /* calculation of the pixel coordinates in the room coordinates */
                xyz[0] = (double) dose_vol->offset[0] + ijk[0] * dose_vol->spacing[0];
                xyz[1] = (double) dose_vol->offset[1] + ijk[1] * dose_vol->spacing[1];
                xyz[2] = (double) dose_vol->offset[2] + ijk[2] * dose_vol->spacing[2]; // xyz[3] always = 1.0
				
                sigma_3 = 3 * plan->sigma_vol_lg->get_rgdepth(xyz);
                if (sigma_3 <= 0)
                {
                    continue;
                }
                else
                {
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
							
                            rg_length = plan->rpl_vol->get_rgdepth(xyz_travel);
                            if (rg_length <= 0)
                            {
                                continue;
                            }
                            else
                            {
                                /* the dose from that sector is summed */
                                sigma_travel = plan->sigma_vol->get_rgdepth(xyz_travel);
                                if (sigma_travel <= 0) 
                                {
                                    continue;
                                }
                                else
                                {
                                    central_sector_dose = plan->beam->lookup_sobp_dose((float) rg_length)* (1/(sigma_travel*sqrt(2*M_PI)));
                                    radius = vec3_dist(xyz, xyz_travel);
                                    r_s = radius/sigma_travel;
                                    dose_img[idx] += plan->get_normalization_dose() * plan->beam->get_beamWeight() * central_sector_dose * get_off_axis(r_s) * (*area)[i] * sigma_3 * sigma_3 * ppp->weight / dose_norm; // * is normalized to a radius =1, need to be adapted to a 3_sigma radius circle
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

void compute_dose_ray_shackleford(Volume::Pointer dose_vol, Photon_plan* plan, const Photon_depth_dose* ppp, std::vector<double>* area, std::vector<double>* xy_grid, int radius_sample, int theta_sample, float normalization_dose)
{
    int ijk[3] = {0,0,0};
    double xyz[4] = {0,0,0,1};
    double xyz_travel[4] = {0,0,0,1};
    double tmp_xy[4] = {0,0,0,1};
    double tmp_cst = 0;

    int idx = 0;
	
    int ct_dim[3] = {dose_vol->dim[0], dose_vol->dim[1], dose_vol->dim[2]};
    double vec_ud[4] = {0,0,0,1};
    double vec_rl[4] = {0,0,0,1};

    float* ct_img = (float*) plan->get_patient_volume()->img;
    float* dose_img = (float*) dose_vol->img;

    double sigma_travel = 0;
    double sigma_3 = 0;
    double rg_length = 0;
    double central_sector_dose = 0;
    double radius = 0;
    double theta = 0;
    double r_s = 0;

    vec3_copy(vec_ud, plan->get_aperture()->pdn);
    vec3_normalize1(vec_ud);
    vec3_copy(vec_rl, plan->get_aperture()->prt);
    vec3_normalize1(vec_rl);

    for (ijk[0] = 0; ijk[0] < ct_dim[0]; ijk[0]++){
        for (ijk[1] = 0; ijk[1] < ct_dim[1]; ijk[1]++){
            for (ijk[2] = 0; ijk[2] < ct_dim[2]; ijk[2]++){
                idx = ijk[0] + ct_dim[0] * (ijk[1] + ct_dim[1] * ijk[2]);

                if (ct_img[idx] <= -1000) {continue;} // if this pixel is in the air, no dose delivered

                /* calculation of the pixel coordinates in the room coordinates */
                xyz[0] = (double) dose_vol->offset[0] + ijk[0] * dose_vol->spacing[0];
                xyz[1] = (double) dose_vol->offset[1] + ijk[1] * dose_vol->spacing[1];
                xyz[2] = (double) dose_vol->offset[2] + ijk[2] * dose_vol->spacing[2]; // xyz[3] always = 1.0
				
                sigma_3 = 3 * plan->sigma_vol_lg->get_rgdepth(xyz);
                if (sigma_3 <= 0)
                {
                    continue;
                }
                else
                {
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
							
                            rg_length = plan->rpl_vol->get_rgdepth(xyz_travel);
                            if (rg_length <= 0)
                            {
                                continue;
                            }
                            else
                            {
                                /* the dose from that sector is summed */
                                sigma_travel = plan->sigma_vol->get_rgdepth(xyz_travel);
                                if (sigma_travel <= 0) 
                                {
                                    continue;
                                }
                                else
                                {
                                    central_sector_dose = plan->beam->lookup_sobp_dose((float) rg_length)* (1/(sigma_travel*sqrt(2*M_PI)));
                                    radius = vec3_dist(xyz, xyz_travel);
                                    r_s = radius/sigma_travel;
                                    dose_img[idx] += normalization_dose * central_sector_dose * get_off_axis(r_s) * (*area)[i] * sigma_3 * sigma_3 * ppp->weight; // * is normalized to a radius =1, need to be adapted to a 3_sigma radius circle
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

void
calculate_rpl_coordinates_xyz(std::vector<std:: vector<double> >* xyz_coordinates_volume, Rpl_volume* rpl_volume)
{
    double aperture[3] = {0.0,0.0,0.0};
    double entrance[3] = {0.0,0.0,0.0};
    double ray_bev[3] = {0.0,0.0,0.0};
    double vec_antibug_prt[3] = {0.0,0.0,0.0};

    int dim[3] = {rpl_volume->get_vol()->dim[0],rpl_volume->get_vol()->dim[1],rpl_volume->get_vol()->dim[2]};
    int idx2d = 0;   
    int idx3d = 0;

    for (int i = 0; i < rpl_volume->get_vol()->dim[0];i++){
        for (int j = 0; j < rpl_volume->get_vol()->dim[1];j++){
        
            idx2d = j * dim[0] + i;
            Ray_data* ray_data = &rpl_volume->get_Ray_data()[idx2d];

            vec3_cross(vec_antibug_prt, rpl_volume->get_aperture()->pdn, rpl_volume->get_proj_volume()->get_nrm());
            ray_bev[0] = vec3_dot(ray_data->ray, vec_antibug_prt);
            ray_bev[1] = vec3_dot(ray_data->ray, rpl_volume->get_aperture()->pdn);
            ray_bev[2] = -vec3_dot(ray_data->ray, rpl_volume->get_proj_volume()->get_nrm()); // ray_beam_eye_view is already normalized

            find_xyz_center(aperture, ray_bev, rpl_volume->get_aperture()->get_distance(),0);
            find_xyz_center_entrance(entrance, ray_bev, rpl_volume->get_front_clipping_plane()-rpl_volume->get_aperture()->get_distance());
            vec3_add2(entrance, aperture);

            for (int k = 0; k < rpl_volume->get_vol()->dim[2]; k++){
                idx3d = k*dim[0]*dim[1] + idx2d;
                for (int l = 0; l < 3; l++)
                {
                    (*xyz_coordinates_volume)[idx3d][l] = entrance[l] + (double) k * ray_bev[l];
                }
            }
        }
    }
}

void copy_rpl_density(std::vector<double>* CT_density_vol, Rpl_volume* rpl_dose_volume)
{
    float* img = (float*) rpl_dose_volume->get_vol()->img;

    for (int i = 0; i < rpl_dose_volume->get_vol()->dim[0] * rpl_dose_volume->get_vol()->dim[1] * rpl_dose_volume->get_vol()->dim[2]; i++)
    {
        (*CT_density_vol)[i] = img[i];
    }
}

void build_hong_grid(std::vector<double>* area, std::vector<double>* xy_grid, int radius_sample, int theta_sample)
{
    double dr = 1.0 / (double) radius_sample;
    double dt = 2.0 * M_PI / (double) theta_sample;

    for (int i = 0; i < radius_sample; i++)
    {
        (*area)[i] = M_PI * dr * dr * ( 2 * i + 1 ) / (double) theta_sample; // [(i+1)^2 - i^2] * dr^2

        for (int j = 0; j < theta_sample; j++)
        {
            (*xy_grid)[2*(i*theta_sample+j)] = ((double) i + 0.5)* dr * sin ((double) j * dt);
            (*xy_grid)[2*(i*theta_sample+j)+1] = ((double) i + 0.5) * dr * cos ((double) j * dt);
        }
    }	
}

void 
dose_volume_reconstruction (
    Rpl_volume* rpl_dose_vol, 
    Volume::Pointer dose_vol, 
    Ion_plan* plan
)
{
    /* scan through patient CT Volume */
    plm_long ct_ijk[3];
    double ct_xyz[4];
    plm_long idx = 0;
    double dose = 0;

    float* dose_img = (float*) dose_vol->img;
    float* ct_img = (float*) plan->get_patient_volume()->img;

    for (ct_ijk[2] = 0; ct_ijk[2] < dose_vol->dim[2]; ct_ijk[2]++) {
        for (ct_ijk[1] = 0; ct_ijk[1] < dose_vol->dim[1]; ct_ijk[1]++) {
            for (ct_ijk[0] = 0; ct_ijk[0] < dose_vol->dim[0]; ct_ijk[0]++) {
                dose = 0.0;

                /* Transform vol index into space coords */
                ct_xyz[0] = (double) (dose_vol->offset[0] + ct_ijk[0] * dose_vol->spacing[0]);
                ct_xyz[1] = (double) (dose_vol->offset[1] + ct_ijk[1] * dose_vol->spacing[1]);
                ct_xyz[2] = (double) (dose_vol->offset[2] + ct_ijk[2] * dose_vol->spacing[2]);
                ct_xyz[3] = (double) 1.0;
                idx = volume_index (dose_vol->dim, ct_ijk);

#if defined (commentout)
                /* This causes strange artifacts in the dose */
                if (ct_img[idx] <= -1000) // if air, no dose
                {
                    continue;
                }
#endif

                dose = plan->rpl_dose_vol->get_rgdepth(ct_xyz);
                if (dose <= 0) {
                    continue;
                }

                /* Insert the dose into the dose volume */
                dose_img[idx] += dose;
            }
        }
    }
}

void dose_volume_reconstruction(Rpl_volume* rpl_dose_vol, Volume::Pointer dose_vol, Photon_plan* plan)
{
    /* scan through patient CT Volume */
    plm_long ct_ijk[3];
    double ct_xyz[4];
    plm_long idx = 0;
    double dose = 0;

    float* dose_img = (float*) dose_vol->img;
    float* ct_img = (float*) plan->get_patient_volume()->img;

    for (ct_ijk[2] = 0; ct_ijk[2] < dose_vol->dim[2]; ct_ijk[2]++) {
        for (ct_ijk[1] = 0; ct_ijk[1] < dose_vol->dim[1]; ct_ijk[1]++) {
            for (ct_ijk[0] = 0; ct_ijk[0] < dose_vol->dim[0]; ct_ijk[0]++) {
                dose = 0.0;

                /* Transform vol index into space coords */
                ct_xyz[0] = (double) (dose_vol->offset[0] + ct_ijk[0] * dose_vol->spacing[0]);
                ct_xyz[1] = (double) (dose_vol->offset[1] + ct_ijk[1] * dose_vol->spacing[1]);
                ct_xyz[2] = (double) (dose_vol->offset[2] + ct_ijk[2] * dose_vol->spacing[2]);
                ct_xyz[3] = (double) 1.0;
                idx = volume_index (dose_vol->dim, ct_ijk);

                if (ct_img[idx] <= -1000) // if air, no dose
                {
                    continue;
                }
                else
                {
                    dose = plan->rpl_dose_vol->get_rgdepth(ct_xyz);
                    if (dose <=0){continue;}
                }

                /* Insert the dose into the dose volume */
                dose_img[idx] += dose;
            }
        }
    }
}

double get_dose_norm(char flavor, double energy, double PB_density)
{
    if (flavor == 'a')
    {
        return 1; // to be defined
    }
    else if (flavor == 'f')
    {
        return PB_density * (30.5363 + 0.21570 * energy - 0.003356 * energy * energy + 0.00000917 * energy * energy * energy);
    }
    else if (flavor == 'g')
    {
        return PB_density * (1470.843 - 8.43943 * energy - 0.005703 * energy * energy + 0.000076755 * energy * energy * energy);
    }
    else if (flavor == 'h')
    {
        return 71.5177 - 0.41466 * energy + 0.014798 * energy*energy -0.00004280 * energy*energy*energy;
    }
    else
    {
        return 1;
    }
}

void 
find_xyz_center(double* xyz_ray_center, double* ray, float z_axis_offset, int k)
{
    float alpha = 0.0f;

    xyz_ray_center[2] = z_axis_offset+(double)k;

    alpha = xyz_ray_center[2] /(double) ray[2];
    xyz_ray_center[0] = alpha * ray[0];
    xyz_ray_center[1] = alpha * ray[1];
    return;
}


void 
find_xyz_center_entrance(double* xyz_ray_center, double* ray, float z_axis_offset)
{
    xyz_ray_center[0] = z_axis_offset * ray[0];
    xyz_ray_center[1] = z_axis_offset * ray[1];
    xyz_ray_center[2] = z_axis_offset * ray[2];

    return;
}

void 
find_ijk_pixel(int* ijk_idx, double* xyz_ray_center, Volume* dose_volume)
{
    ijk_idx[0] = (int) floor((xyz_ray_center[0] - dose_volume->offset[0]) / dose_volume->spacing[0] + 0.5);
    ijk_idx[1] = (int) floor((xyz_ray_center[1] - dose_volume->offset[1]) / dose_volume->spacing[1] + 0.5);
    ijk_idx[2] = (int) floor((xyz_ray_center[2] - dose_volume->offset[2]) / dose_volume->spacing[2] + 0.5);
}

void 
find_xyz_from_ijk(double* xyz, Volume* volume, int* ijk)
{
    xyz[0] = volume->offset[0] + ijk[0]*volume->spacing[0];
    xyz[1] = volume->offset[1] + ijk[1]*volume->spacing[1];
    xyz[2] = volume->offset[2] + ijk[2]*volume->spacing[2];
}

double distance(const std::vector< std::vector<double> >& v, int i1, int i2)
{
    return sqrt( (v[i1][0]-v[i2][0])*(v[i1][0]-v[i2][0]) + (v[i1][1]-v[i2][1])*(v[i1][1]-v[i2][1]) + (v[i1][2]-v[i2][2])*(v[i1][2]-v[i2][2]) );
}

double double_gaussian_interpolation(double* gaussian_center, double* pixel_center, double sigma, double* spacing)
{
    double x1 = pixel_center[0] - 0.5 * spacing[0];
    double x2 = x1 + spacing[0];
    double y1 = pixel_center[1] - 0.5 * spacing[1];
    double y2 = y1 + spacing[1];

    double z = .25 
        * (erf_gauss((x2-gaussian_center[0])/(sigma*1.4142135)) - erf_gauss((x1-gaussian_center[0])/(sigma*1.4142135)))
        * (erf_gauss((y2-gaussian_center[1])/(sigma*1.4142135)) - erf_gauss((y1-gaussian_center[1])/(sigma*1.4142135)));
    return z;
}

double erf_gauss(double x)
{
    /* constant */
    double a1 =  0.254829592;
    double a2 = -0.284496736;
    double a3 =  1.421413741;
    double a4 = -1.453152027;
    double a5 =  1.061405429;
    double p  =  0.3275911;

    /* save the sign of x */
    int sign = 1;
    if (x < 0) {sign = -1;}
    x = fabs(x);

    /* erf interpolation */
    double t = 1.0/(1.0 + p*x);
    double y = 1.0 - (((((a5*t + a4)*t) + a3)*t + a2)*t + a1)*t*exp(-x*x);

    return sign*y;
}

float LR_interpolation(float density)
{
    return 36.08f*pow(density,-1.548765f); // in cm
}

float WER_interpolation(float density) // interpolation between adip, water, muscle, PMMA and bone
{
    if (density <=1)
    {
        return 0.3825f * density + .6175f;
    }
    else if (density > 1 && density <=1.04)
    {
        return .275f * density + .725f;
    }
    else if (density > 1.04 && density <= 1.19)
    {
        return .1047f * density + .9021f;
    }
    else
    {
        return .0803f * density + .9311f;
    }
}

double getrange(double energy)
{
    double energy1 = 0;
    double energy2 = 0;
    double range1 = 0;
    double range2 = 0;
    int i=0;

    if (energy >0)
    {
	while (energy >= energy1)
	{
            energy1 = lookup_range_water[i][0];
	    range1 = lookup_range_water[i][1];

	    if (energy >= energy1)
	    {
	    	energy2 = energy1;
		range2 = range1;
	    }
	    i++;
	}
	return (range2+(energy-energy2)*(range1-range2)/(energy1-energy2));
    }
    else
    {
	return 0;
    }
}

double getstop (double energy)
{
    /* GCS FIX: It should be possible to simply march along LUT rather than 
       searching at every step */
    int i_lo = 0, i_hi = 131;
    double energy_lo = lookup_stop_water[i_lo][0];
    double stop_lo = lookup_stop_water[i_lo][1];
    double energy_hi = lookup_stop_water[i_hi][0];
    double stop_hi = lookup_stop_water[i_hi][1];

    if (energy <= energy_lo) {
        return stop_lo;
    }
    if (energy >= energy_hi) {
        return stop_hi;
    }

    /* Use binary search to find lookup table entries */
    for (int dif = i_hi - i_lo; dif > 1; dif = i_hi - i_lo) {
        int i_test = i_lo + ((dif + 1) / 2);
        double energy_test = lookup_stop_water[i_test][0];
        if (energy > energy_test) {
            energy_lo = energy_test;
            i_lo = i_test;
        } else {
            energy_hi = energy_test;
            i_hi = i_test;
        }
    }

    stop_lo = lookup_stop_water[i_lo][1];
    stop_hi = lookup_stop_water[i_hi][1];
    return stop_lo + 
        (energy-energy_lo) * (stop_hi-stop_lo) / (energy_hi-energy_lo);
}

double get_off_axis(double r_s)
{
    double r_s_1 = 0;
    double r_s_2 = 0;
    double off_axis1 = 0;
    double off_axis2 = 0;

    int i=0;

    if (r_s >0)
    {
        while (r_s >= r_s_1)
        {
            r_s_1 = lookup_off_axis[i][0];
            off_axis1 = lookup_off_axis[i][1];

            if (r_s >= r_s_1)
            {
                r_s_2 = r_s_1;
                off_axis2 = off_axis1;
            }
            i++;
        }
        return (off_axis2+(r_s-r_s_2)*(off_axis1-off_axis2)/(r_s_1-r_s_2));
    }
    else
    {
        return 0;
    }
}

const double lookup_range_water[][2] ={
1.000E-03,	6.319E-06,
1.500E-03,	8.969E-06,	
2.000E-03,	1.137E-05,	
2.500E-03,	1.357E-05,	
3.000E-03,	1.560E-05,	
4.000E-03,	1.930E-05,	
5.000E-03,	2.262E-05,	
6.000E-03,	2.567E-05,	
7.000E-03,	2.849E-05,	
8.000E-03,	3.113E-05,	
9.000E-03,	3.363E-05,	
1.000E-02,	3.599E-05,	
1.250E-02,	4.150E-05,	
1.500E-02,	4.657E-05,	
1.750E-02,	5.131E-05,	
2.000E-02,	5.578E-05,	
2.250E-02,	6.005E-05,	
2.500E-02,	6.413E-05,	
2.750E-02,	6.806E-05,	
3.000E-02,	7.187E-05,	
3.500E-02,	7.916E-05,	
4.000E-02,	8.613E-05,	
4.500E-02,	9.284E-05,	
5.000E-02,	9.935E-05,	
5.500E-02,	1.057E-04,	
6.000E-02,	1.120E-04,	
6.500E-02,	1.182E-04,	
7.000E-02,	1.243E-04,	
7.500E-02,	1.303E-04,	
8.000E-02,	1.364E-04,	
8.500E-02,	1.425E-04,	
9.000E-02,	1.485E-04,	
9.500E-02,	1.546E-04,	
1.000E-01,	1.607E-04,	
1.250E-01,	1.920E-04,	
1.500E-01,	2.249E-04,	
1.750E-01,	2.598E-04,	
2.000E-01,	2.966E-04,	
2.250E-01,	3.354E-04,	
2.500E-01,	3.761E-04,	
2.750E-01,	4.186E-04,	
3.000E-01,	4.631E-04,	
3.500E-01,	5.577E-04,	
4.000E-01,	6.599E-04,	
4.500E-01,	7.697E-04,	
5.000E-01,	8.869E-04,	
5.500E-01,	1.012E-03,	
6.000E-01,	1.144E-03,	
6.500E-01,	1.283E-03,	
7.000E-01,	1.430E-03,	
7.500E-01,	1.584E-03,	
8.000E-01,	1.745E-03,	
8.500E-01,	1.913E-03,	
9.000E-01,	2.088E-03,	
9.500E-01,	2.270E-03,	
1.000E+00,	2.458E-03,	
1.250E+00,	3.499E-03,	
1.500E+00,	4.698E-03,	
1.750E+00,	6.052E-03,	
2.000E+00,	7.555E-03,	
2.250E+00,	9.203E-03,	
2.500E+00,	1.099E-02,	
2.750E+00,	1.292E-02,	
3.000E+00,	1.499E-02,	
3.500E+00,	1.952E-02,	
4.000E+00,	2.458E-02,	
4.500E+00,	3.015E-02,	
5.000E+00,	3.623E-02,	
5.500E+00,	4.279E-02,	
6.000E+00,	4.984E-02,	
6.500E+00,	5.737E-02,	
7.000E+00,	6.537E-02,	
7.500E+00,	7.384E-02,	
8.000E+00,	8.277E-02,	
8.500E+00,	9.215E-02,	
9.000E+00,	1.020E-01,	
9.500E+00,	1.123E-01,	
1.000E+01,	1.230E-01,	
1.250E+01,	1.832E-01,	
1.500E+01,	2.539E-01,	
1.750E+01,	3.350E-01,	
2.000E+01,	4.260E-01,	
2.500E+01,	6.370E-01,	
2.750E+01,	7.566E-01,	
3.000E+01,	8.853E-01,	
3.500E+01,	1.170E+00,	
4.000E+01,	1.489E+00,	
4.500E+01,	1.841E+00,	
5.000E+01,	2.227E+00,	
5.500E+01,	2.644E+00,	
6.000E+01,	3.093E+00,	
6.500E+01,	3.572E+00,	
7.000E+01,	4.080E+00,	
7.500E+01,	4.618E+00,	
8.000E+01,	5.184E+00,	
8.500E+01,	5.777E+00,	
9.000E+01,	6.398E+00,	
9.500E+01,	7.045E+00,	
1.000E+02,	7.718E+00,	
1.250E+02,	1.146E+01,	
1.500E+02,	1.577E+01,	
1.750E+02,	2.062E+01,	
2.000E+02,	2.596E+01,	
2.250E+02,	3.174E+01,	
2.500E+02,	3.794E+01,	
2.750E+02,	4.452E+01,	
3.000E+02,	5.145E+01,	
3.500E+02,	6.628E+01,	
4.000E+02,	8.225E+01,	
4.500E+02,	9.921E+01,	
5.000E+02,	1.170E+02,	
5.500E+02,	1.356E+02,	
6.000E+02,	1.549E+02,	
6.500E+02,	1.747E+02,	
7.000E+02,	1.951E+02,	
7.500E+02,	2.159E+02,	
8.000E+02,	2.372E+02,	
8.500E+02,	2.588E+02,	
9.000E+02,	2.807E+02,	
9.500E+02,	3.029E+02,	
1.000E+03,	3.254E+02,	
1.500E+03,	5.605E+02,	
2.000E+03,	8.054E+02,	
2.500E+03,	1.054E+03,	
3.000E+03,	1.304E+03,	
4.000E+03,	1.802E+03,	
5.000E+03,	2.297E+03,	
6.000E+03,	2.787E+03,	
7.000E+03,	3.272E+03,	
8.000E+03,	3.752E+03,	
9.000E+03,	4.228E+03,	
1.000E+04,	4.700E+03,	
};

/* This table has 132 entries */
const double lookup_stop_water[][2] =
{
0.001,	176.9,
0.0015,	198.4,
0.002,	218.4,
0.0025,	237,
0.003,	254.4,
0.004,	286.4,
0.005,	315.3,
0.006,	342,
0.007,	366.7,
0.008,	390,
0.009,	412,
0.01,	432.9,
0.0125,	474.5,
0.015,	511,
0.0175,	543.7,
0.02,	573.3,
0.0225,	600.1,
0.025,	624.5,
0.0275,	646.7,
0.03,	667.1,
0.035,	702.8,
0.04,	732.4,
0.045,	756.9,
0.05,	776.8,
0.055,	792.7,
0.06,	805,
0.065,	814.2,
0.07,	820.5,
0.075,	824.3,
0.08,	826,
0.085,	825.8,
0.09,	823.9,
0.095,	820.6,
0.1,	816.1,
0.125,	781.4,
0.15,	737.1,
0.175,	696.9,
0.2,	661.3,
0.225,	629.4,
0.25,	600.6,
0.275,	574.4,
0.3,	550.4,
0.35,	508,
0.4,	471.9,
0.45,	440.6,
0.5,	413.2,
0.55,	389.1,
0.6,	368,
0.65,	349.2,
0.7,	332.5,
0.75,	317.5,
0.8,	303.9,
0.85,	291.7,
0.9,	280.5,
0.95,	270.2,
1,	260.8,
1.25,	222.9,
1.5,	195.7,
1.75,	174.9,
2,	158.6,
2.25,	145.4,
2.5,	134.4,
2.75,	125.1,
3,	117.2,
3.5,	104.2,
4,	94.04,
4.5,	85.86,
5,	79.11,
5.5,	73.43,
6,	68.58,
6.5,	64.38,
7,	60.71,
7.5,	57.47,
8,	54.6,
8.5,	52.02,
9,	49.69,
9.5,	47.59,
10,	45.67,
12.5,	38.15,
15,	32.92,
17.5,	29.05,
20,	26.07,
25,	21.75,
27.5,	20.13,
30,	18.76,
35,	16.56,
40,	14.88,
45,	13.54,
50,	12.45,
55,	11.54,
60,	10.78,
65,	10.13,
70,	9.559,
75,	9.063,
80,	8.625,
85,	8.236,
90,	7.888,
95,	7.573,
100,	7.289,
125,	6.192,
150,	5.445,
175,	4.903,
200,	4.492,
225,	4.17,
250,	3.911,
275,	3.698,
300,	3.52,
350,	3.241,
400,	3.032,
450,	2.871,
500,	2.743,
550,	2.64,
600,	2.556,
650,	2.485,
700,	2.426,
750,	2.376,
800,	2.333,
850,	2.296,
900,	2.264,
950,	2.236,
1000,	2.211,
1500,	2.07,
2000,	2.021,
2500,	2.004,
3000,	2.001,
4000,	2.012,
5000,	2.031,
6000,	2.052,
7000,	2.072,
8000,	2.091,
9000,	2.109,
10000,	2.126,
};

const double lookup_off_axis[][2]={ // x[0] = r/sigma, x[1] = exp(-x²/(2 * sigma²))
0.00, 1.0000,
0.01, 1.0000,
0.02, 0.9998,
0.03, 0.9996,
0.04, 0.9992,
0.05, 0.9988,
0.06, 0.9982,
0.07, 0.9976,
0.08, 0.9968,
0.09, 0.9960,
0.10, 0.9950,
0.11, 0.9940,
0.12, 0.9928,
0.13, 0.9916,
0.14, 0.9902,
0.15, 0.9888,
0.16, 0.9873,
0.17, 0.9857,
0.18, 0.9839,
0.19, 0.9821,
0.20, 0.9802,
0.21, 0.9782,
0.22, 0.9761,
0.23, 0.9739,
0.24, 0.9716,
0.25, 0.9692,
0.26, 0.9668,
0.27, 0.9642,
0.28, 0.9616,
0.29, 0.9588,
0.30, 0.9560,
0.31, 0.9531,
0.32, 0.9501,
0.33, 0.9470,
0.34, 0.9438,
0.35, 0.9406,
0.36, 0.9373,
0.37, 0.9338,
0.38, 0.9303,
0.39, 0.9268,
0.40, 0.9231,
0.41, 0.9194,
0.42, 0.9156,
0.43, 0.9117,
0.44, 0.9077,
0.45, 0.9037,
0.46, 0.8996,
0.47, 0.8954,
0.48, 0.8912,
0.49, 0.8869,
0.50, 0.8825,
0.51, 0.8781,
0.52, 0.8735,
0.53, 0.8690,
0.54, 0.8643,
0.55, 0.8596,
0.56, 0.8549,
0.57, 0.8501,
0.58, 0.8452,
0.59, 0.8403,
0.60, 0.8353,
0.61, 0.8302,
0.62, 0.8251,
0.63, 0.8200,
0.64, 0.8148,
0.65, 0.8096,
0.66, 0.8043,
0.67, 0.7990,
0.68, 0.7936,
0.69, 0.7882,
0.70, 0.7827,
0.71, 0.7772,
0.72, 0.7717,
0.73, 0.7661,
0.74, 0.7605,
0.75, 0.7548,
0.76, 0.7492,
0.77, 0.7435,
0.78, 0.7377,
0.79, 0.7319,
0.80, 0.7261,
0.81, 0.7203,
0.82, 0.7145,
0.83, 0.7086,
0.84, 0.7027,
0.85, 0.6968,
0.86, 0.6909,
0.87, 0.6849,
0.88, 0.6790,
0.89, 0.6730,
0.90, 0.6670,
0.91, 0.6610,
0.92, 0.6549,
0.93, 0.6489,
0.94, 0.6429,
0.95, 0.6368,
0.96, 0.6308,
0.97, 0.6247,
0.98, 0.6187,
0.99, 0.6126,
1.00, 0.6065,
1.01, 0.6005,
1.02, 0.5944,
1.03, 0.5883,
1.04, 0.5823,
1.05, 0.5762,
1.06, 0.5702,
1.07, 0.5641,
1.08, 0.5581,
1.09, 0.5521,
1.10, 0.5461,
1.11, 0.5401,
1.12, 0.5341,
1.13, 0.5281,
1.14, 0.5222,
1.15, 0.5162,
1.16, 0.5103,
1.17, 0.5044,
1.18, 0.4985,
1.19, 0.4926,
1.20, 0.4868,
1.21, 0.4809,
1.22, 0.4751,
1.23, 0.4693,
1.24, 0.4636,
1.25, 0.4578,
1.26, 0.4521,
1.27, 0.4464,
1.28, 0.4408,
1.29, 0.4352,
1.30, 0.4296,
1.31, 0.4240,
1.32, 0.4184,
1.33, 0.4129,
1.34, 0.4075,
1.35, 0.4020,
1.36, 0.3966,
1.37, 0.3912,
1.38, 0.3859,
1.39, 0.3806,
1.40, 0.3753,
1.41, 0.3701,
1.42, 0.3649,
1.43, 0.3597,
1.44, 0.3546,
1.45, 0.3495,
1.46, 0.3445,
1.47, 0.3394,
1.48, 0.3345,
1.49, 0.3295,
1.50, 0.3247,
1.51, 0.3198,
1.52, 0.3150,
1.53, 0.3102,
1.54, 0.3055,
1.55, 0.3008,
1.56, 0.2962,
1.57, 0.2916,
1.58, 0.2870,
1.59, 0.2825,
1.60, 0.2780,
1.61, 0.2736,
1.62, 0.2692,
1.63, 0.2649,
1.64, 0.2606,
1.65, 0.2563,
1.66, 0.2521,
1.67, 0.2480,
1.68, 0.2439,
1.69, 0.2398,
1.70, 0.2357,
1.71, 0.2318,
1.72, 0.2278,
1.73, 0.2239,
1.74, 0.2201,
1.75, 0.2163,
1.76, 0.2125,
1.77, 0.2088,
1.78, 0.2051,
1.79, 0.2015,
1.80, 0.1979,
1.81, 0.1944,
1.82, 0.1909,
1.83, 0.1874,
1.84, 0.1840,
1.85, 0.1806,
1.86, 0.1773,
1.87, 0.1740,
1.88, 0.1708,
1.89, 0.1676,
1.90, 0.1645,
1.91, 0.1614,
1.92, 0.1583,
1.93, 0.1553,
1.94, 0.1523,
1.95, 0.1494,
1.96, 0.1465,
1.97, 0.1436,
1.98, 0.1408,
1.99, 0.1381,
2.00, 0.1353,
2.01, 0.1326,
2.02, 0.1300,
2.03, 0.1274,
2.04, 0.1248,
2.05, 0.1223,
2.06, 0.1198,
2.07, 0.1174,
2.08, 0.1150,
2.09, 0.1126,
2.10, 0.1103,
2.11, 0.1080,
2.12, 0.1057,
2.13, 0.1035,
2.14, 0.1013,
2.15, 0.0991,
2.16, 0.0970,
2.17, 0.0949,
2.18, 0.0929,
2.19, 0.0909,
2.20, 0.0889,
2.21, 0.0870,
2.22, 0.0851,
2.23, 0.0832,
2.24, 0.0814,
2.25, 0.0796,
2.26, 0.0778,
2.27, 0.0760,
2.28, 0.0743,
2.29, 0.0727,
2.30, 0.0710,
2.31, 0.0694,
2.32, 0.0678,
2.33, 0.0662,
2.34, 0.0647,
2.35, 0.0632,
2.36, 0.0617,
2.37, 0.0603,
2.38, 0.0589,
2.39, 0.0575,
2.40, 0.0561,
2.41, 0.0548,
2.42, 0.0535,
2.43, 0.0522,
2.44, 0.0510,
2.45, 0.0497,
2.46, 0.0485,
2.47, 0.0473,
2.48, 0.0462,
2.49, 0.0450,
2.50, 0.0439,
2.51, 0.0428,
2.52, 0.0418,
2.53, 0.0407,
2.54, 0.0397,
2.55, 0.0387,
2.56, 0.0377,
2.57, 0.0368,
2.58, 0.0359,
2.59, 0.0349,
2.60, 0.0340,
2.61, 0.0332,
2.62, 0.0323,
2.63, 0.0315,
2.64, 0.0307,
2.65, 0.0299,
2.66, 0.0291,
2.67, 0.0283,
2.68, 0.0276,
2.69, 0.0268,
2.70, 0.0261,
2.71, 0.0254,
2.72, 0.0247,
2.73, 0.0241,
2.74, 0.0234,
2.75, 0.0228,
2.76, 0.0222,
2.77, 0.0216,
2.78, 0.0210,
2.79, 0.0204,
2.80, 0.0198,
2.81, 0.0193,
2.82, 0.0188,
2.83, 0.0182,
2.84, 0.0177,
2.85, 0.0172,
2.86, 0.0167,
2.87, 0.0163,
2.88, 0.0158,
2.89, 0.0154,
2.90, 0.0149,
2.91, 0.0145,
2.92, 0.0141,
2.93, 0.0137,
2.94, 0.0133,
2.95, 0.0129,
2.96, 0.0125,
2.97, 0.0121,
2.98, 0.0118,
2.99, 0.0114,
3.00, 0.0111,
};
