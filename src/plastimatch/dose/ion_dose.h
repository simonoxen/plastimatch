/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _ion_dose_h_
#define _ion_dose_h_

#include "plmdose_config.h"
#include "plm_image.h"
#include "ion_plan.h"

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

#endif
