/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _sigma_spread_h_
#define _sigma_spread_h_

#include "plmdose_config.h"
#include <math.h>
#include <vector>
#include "ion_plan.h"
#include "ion_pristine_peak.h"
#include "rpl_volume.h"
#include "photon_plan.h"
#include "photon_depth_dose.h"
#include "proj_volume.h"
#include "volume.h"

extern const double lookup_range_water[][2];
extern const double lookup_stop_water[][2];
extern const double lookup_off_axis[][2];

class Ion_beam;

void convert_radiologic_length_to_sigma(Ion_plan* ion_plan, float energy, float* sigma_max); // compute the sigma_vol and return sigma_max
void convert_radiologic_length_to_sigma_lg(Ion_plan* ion_plan, float energy, float* sigma_max); // compute the sigma_vol and return sigma_max for larger volumes
void convert_radiologic_length_to_sigma(Photon_plan* ion_plan, float energy, float* sigma_max); // compute the sigma_vol and return sigma_max
void convert_radiologic_length_to_sigma_lg(Photon_plan* ion_plan, float energy, float* sigma_max); // compute the sigma_vol and return sigma_max for larger volumes

void length_to_sigma(std::vector<float>* p_sigma, std::vector<float>* p_density, float spacing_z,float* sigma_max, float energy, float source_size);
void length_to_sigma_photon(std::vector<float>* p_sigma, std::vector<float>* p_density, float spacing_z,float* sigma_max, float energy, float source_size);

void compute_dose_ray_desplanques(Volume* dose_volume, Volume::Pointer ct_vol, Rpl_volume* rpl_vol, Rpl_volume* sigma_vol, Rpl_volume* ct_vol_density, Ion_beam* beam, Volume::Pointer final_dose_volume, const Ion_pristine_peak* ppp);
void compute_dose_ray_desplanques(Volume* dose_volume, Volume::Pointer ct_vol, Rpl_volume* rpl_vol, Rpl_volume* sigma_vol, Rpl_volume* ct_vol_density, Photon_beam* beam, Volume::Pointer final_dose_volume, const Photon_depth_dose* ppp);

void compute_dose_ray_sharp(Volume::Pointer ct_vol, Rpl_volume* rpl_vol, Rpl_volume* sigma_vol, Rpl_volume* ct_vol_density, Ion_beam* beam, Rpl_volume* rpl_dose_volume, Aperture::Pointer ap, const Ion_pristine_peak* ppp, int* margins);
void compute_dose_ray_sharp(Volume::Pointer ct_vol, Rpl_volume* rpl_vol, Rpl_volume* sigma_vol, Rpl_volume* ct_vol_density, Photon_beam* beam, Rpl_volume* rpl_dose_volume, Aperture::Pointer ap, const Photon_depth_dose* ppp, int* margins);

void compute_dose_ray_shackleford(Volume::Pointer dose_volume, Ion_plan* plan, const Ion_pristine_peak* ppp, std::vector<double>* area, std::vector<double>* xy_grid, int radius_sample, int theta_sample);
void compute_dose_ray_shackleford(Volume::Pointer dose_volume, Photon_plan* plan, const Photon_depth_dose* ppp, std::vector<double>* area, std::vector<double>* xy_grid, int radius_sample, int theta_sample);


void calculate_rpl_coordinates_xyz(std::vector<std::vector<double> >* xyz_coordinates_volume, Rpl_volume* rpl_volume);
void copy_rpl_density(std::vector<double>* CT_density_vol, Rpl_volume* rpl_dose_volume);
void dose_volume_reconstruction(Rpl_volume* rpl_dose_vol,Volume::Pointer dose_vol, Ion_plan* plan);
void dose_volume_reconstruction(Rpl_volume* rpl_dose_vol,Volume::Pointer dose_vol, Photon_plan* plan);

void build_hong_grid(std::vector<double>* area, std::vector<double>* xy_grid, int radius_sample, int theta_sample);

void find_ijk_pixel(int* ijk_idx, double* xyz_ray_center, Volume* dose_volume);
void find_xyz_center_entrance(double* xyz_ray_center, double* ray, float z_axis_offset);
void find_xyz_center(double* xyz_ray_center, double* ray, float z_axis_offset, int k);
void find_xyz_from_ijk(double* xyz, Volume* volume, int* ijk);
double distance(const std::vector< std::vector <double> >&, int, int);
double double_gaussian_interpolation(double* gaussian_center, double* pixel_center, double sigma, double* spacing);
double erf_gauss(double x);

float LR_interpolation(float density);
float WER_interpolation(float density);

double getrange(double energy);
double getstop(double energy);
double get_off_axis(double r2_s2);

#endif
