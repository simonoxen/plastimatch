/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _photon_dose_h_
#define _photon_dose_h_

#include "plmdose_config.h"
#include "plm_image.h"
#include "photon_plan.h"

PLMDOSE_API
Plm_image::Pointer
photon_dose_compute (Photon_plan::Pointer& scene);

double
dose_direct (
    double* ct_xyz,             /* voxel to dose */
    const Photon_plan* scene
);
double
dose_debug (
    double* ct_xyz,             /* voxel to dose */
    const Photon_plan* scene
);
double
dose_scatter (
    double* ct_xyz,
    plm_long* ct_ijk,            // DEBUG
    const Photon_plan* scene
);
double
dose_hong (
    double* ct_xyz,
    plm_long* ct_ijk,            // DEBUG
    const Photon_plan* scene
);

double
dose_hong_maxime (
    double* ct_xyz,
    plm_long* ct_ijk,            // DEBUG
    const Photon_plan* scene
);

double
dose_hong_sharp (
    double* ct_xyz,             /* voxel to dose */
    const Photon_plan* scene
);

#endif
