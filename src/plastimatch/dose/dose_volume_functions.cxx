/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */

#include "dose_volume_functions.h"
#include "proj_volume.h"
#include "ray_data.h"

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

            find_xyz_center(aperture, ray_bev, rpl_volume->get_aperture()->get_distance(),0, rpl_volume->get_vol()->spacing[2]);
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

void 
dose_volume_reconstruction (
    Rpl_volume* rpl_dose_vol, 
    Volume::Pointer dose_vol
)
{
    /* scan through patient CT Volume */
    plm_long ct_ijk[3];
    double ct_xyz[4];
    plm_long idx = 0;
    double dose = 0;

    float* dose_img = (float*) dose_vol->img;

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

                dose = rpl_dose_vol->get_rgdepth(ct_xyz);
                if (dose <= 0) {
                    continue;
                }

                /* Insert the dose into the dose volume */
                dose_img[idx] += dose;
            }
        }
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
find_ijk_pixel(int* ijk_idx, double* xyz_ray_center, Volume* dose_volume)
{
    ijk_idx[0] = (int) floor((xyz_ray_center[0] - dose_volume->offset[0]) / dose_volume->spacing[0] + 0.5);
    ijk_idx[1] = (int) floor((xyz_ray_center[1] - dose_volume->offset[1]) / dose_volume->spacing[1] + 0.5);
    ijk_idx[2] = (int) floor((xyz_ray_center[2] - dose_volume->offset[2]) / dose_volume->spacing[2] + 0.5);
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
find_xyz_center(double* xyz_ray_center, double* ray, float z_axis_offset, int k, float z_spacing)
{
    float alpha = 0.0f;

    xyz_ray_center[2] = z_axis_offset+(double)k * z_spacing;

    alpha = xyz_ray_center[2] /(double) ray[2];
    xyz_ray_center[0] = alpha * ray[0];
    xyz_ray_center[1] = alpha * ray[1];
    return;
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

double double_gaussian_interpolation(double* gaussian_center, double* pixel_center, double sigma, double* spacing)
{
    double x1 = pixel_center[0] - 0.5 * spacing[0];
    double x2 = x1 + spacing[0];
    double y1 = pixel_center[1] - 0.5 * spacing[1];
    double y2 = y1 + spacing[1];
    //printf("\n %lg %lg %lg %lg %lg %lg %lg\n", spacing[0], spacing[1], spacing[2] , x1, x2, y1, y2);

    double z = .25 
        * (erf_gauss((x2-gaussian_center[0])/(sigma*1.4142135)) - erf_gauss((x1-gaussian_center[0])/(sigma*1.4142135)))
        * (erf_gauss((y2-gaussian_center[1])/(sigma*1.4142135)) - erf_gauss((y1-gaussian_center[1])/(sigma*1.4142135)));
    return z;
}

double get_off_axis(double radius, double dr, double sigma)
{
    return M_PI / 8.0 * sigma * ( exp(- (radius - dr)*(radius -dr) / (2 * sigma * sigma)) -  exp(- (radius + dr)*(radius + dr) / (2 * sigma * sigma)));
}

