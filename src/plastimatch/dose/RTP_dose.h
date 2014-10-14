/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _RTP_dose_h_
#define _RTP_dose_h_

#include "plmdose_config.h"
#include "plm_image.h"
#include "RTP_plan.h"
#include "RTP_depth_dose.h"


PLMDOSE_API
Plm_image::Pointer
proton_dose_compute (RTP_plan::Pointer& scene);

double
dose_direct (
    double* ct_xyz,             /* voxel to dose */
    const RTP_plan* scene
);
double
dose_debug (
    double* ct_xyz,             /* voxel to dose */
    const RTP_plan* scene
);
double
dose_scatter (
    double* ct_xyz,
    plm_long* ct_ijk,            // DEBUG
    const RTP_plan* scene
);
double
dose_hong (
    double* ct_xyz,
    plm_long* ct_ijk,            // DEBUG
    const RTP_plan* scene
);

double
dose_hong_maxime (
    double* ct_xyz,
    plm_long* ct_ijk,            // DEBUG
    const RTP_plan* scene
);

double
dose_hong_sharp (
    double* ct_xyz,             /* voxel to dose */
    const RTP_plan* scene
);

void compute_dose_ray_desplanques (
    Volume* dose_volume, 
    Volume::Pointer ct_vol, 
    Rpl_volume* rpl_vol, 
    Rpl_volume* sigma_vol, 
    Rpl_volume* ct_vol_density, 
    RTP_beam* beam, 
    Volume::Pointer final_dose_volume, 
    const RTP_depth_dose* ppp, 
    float normalization_dose
);
void compute_dose_ray_sharp (
    const Volume::Pointer ct_vol, 
    const Rpl_volume* rpl_vol, 
    const Rpl_volume* sigma_vol, 
    const Rpl_volume* ct_vol_density, 
    const RTP_beam* beam, Rpl_volume* rpl_dose_volume, 
    const Aperture::Pointer ap, 
    const RTP_depth_dose* ppp, 
    const int* margins, 
    float normalization_dose
);
void compute_dose_ray_shackleford (
    Volume::Pointer dose_volume, 
    RTP_plan* plan, 
    const RTP_depth_dose* ppp, 
    std::vector<double>* area, 
    std::vector<double>* xy_grid, 
    int radius_sample, 
    int theta_sample
);

double get_dose_norm(char flavor, double energy, double PB_density);

#endif
