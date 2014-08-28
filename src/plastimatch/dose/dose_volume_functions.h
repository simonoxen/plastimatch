/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _dose_volume_functions_h_
#define _dose_volume_functions_h_

#include "rpl_volume.h"

void calculate_rpl_coordinates_xyz(std::vector<std::vector<double> >* xyz_coordinates_volume, Rpl_volume* rpl_volume);
void copy_rpl_density(std::vector<double>* CT_density_vol, Rpl_volume* rpl_dose_volume);
void dose_volume_reconstruction(Rpl_volume* rpl_dose_vol, Volume::Pointer dose_vol);

void build_hong_grid(std::vector<double>* area, std::vector<double>* xy_grid, int radius_sample, int theta_sample);

void find_ijk_pixel(int* ijk_idx, double* xyz_ray_center, Volume* dose_volume);
void find_xyz_center_entrance(double* xyz_ray_center, double* ray, float z_axis_offset);
void find_xyz_center(double* xyz_ray_center, double* ray, float z_axis_offset, int k, float z_spacing);
void find_xyz_from_ijk(double* xyz, Volume* volume, int* ijk);

double distance(const std::vector< std::vector <double> >&, int, int); // Is this useful??

double erf_gauss(double x);
double double_gaussian_interpolation(double* gaussian_center, double* pixel_center, double sigma, double* spacing);

double get_off_axis(double radius, double dr, double sigma);

#endif