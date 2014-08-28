/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _ion_dose_h_
#define _ion_dose_h_

#include "plmdose_config.h"
#include "plm_image.h"
#include "ion_plan.h"
#include "ion_pristine_peak.h"


PLMDOSE_API
Plm_image::Pointer
proton_dose_compute (Ion_plan::Pointer& scene);

double
dose_direct (
    double* ct_xyz,             /* voxel to dose */
    const Ion_plan* scene
);
double
dose_debug (
    double* ct_xyz,             /* voxel to dose */
    const Ion_plan* scene
);
double
dose_scatter (
    double* ct_xyz,
    plm_long* ct_ijk,            // DEBUG
    const Ion_plan* scene
);
double
dose_hong (
    double* ct_xyz,
    plm_long* ct_ijk,            // DEBUG
    const Ion_plan* scene
);

double
dose_hong_maxime (
    double* ct_xyz,
    plm_long* ct_ijk,            // DEBUG
    const Ion_plan* scene
);

double
dose_hong_sharp (
    double* ct_xyz,             /* voxel to dose */
    const Ion_plan* scene
);

void compute_dose_ray_desplanques (
    Volume* dose_volume, 
    Volume::Pointer ct_vol, 
    Rpl_volume* rpl_vol, 
    Rpl_volume* sigma_vol, 
    Rpl_volume* ct_vol_density, 
    Ion_beam* beam, 
    Volume::Pointer final_dose_volume, 
    const Ion_pristine_peak* ppp, 
    float normalization_dose
);
void compute_dose_ray_sharp (
    const Volume::Pointer ct_vol, 
    const Rpl_volume* rpl_vol, 
    const Rpl_volume* sigma_vol, 
    const Rpl_volume* ct_vol_density, 
    const Ion_beam* beam, Rpl_volume* rpl_dose_volume, 
    const Aperture::Pointer ap, 
    const Ion_pristine_peak* ppp, 
    const int* margins, 
    float normalization_dose
);
void compute_dose_ray_shackleford (
    Volume::Pointer dose_volume, 
    Ion_plan* plan, 
    const Ion_pristine_peak* ppp, 
    std::vector<double>* area, 
    std::vector<double>* xy_grid, 
    int radius_sample, 
    int theta_sample
);

double get_dose_norm(char flavor, double energy, double PB_density);

#endif
