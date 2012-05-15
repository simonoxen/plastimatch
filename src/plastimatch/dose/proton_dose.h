/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _proton_dose_h_
#define _proton_dose_h_

#include "plmdose_config.h"
#include "threading.h"

class Volume;

#define INDEX_OF(ijk, dim) \
    (((ijk[2] * dim[1] + ijk[1]) * dim[0]) + ijk[0])

typedef struct proton_dose_parms Proton_dose_parms;
struct proton_dose_parms {
    Threading threading;

    char flavor;                     /* Which algorithm? */
    float src[3];                    /* Beam source */
    float isocenter[3];              /* Beam target */
    float vup[3];                    /* Aperture orientation */

    float scale;
    float ray_step;                  /* Uniform ray step size (mm) */
    char* input_pep_fn;              /* Proton energy profile */
    char* input_fn;
    char* output_fn;
    int debug;

    /* Speed hacks */
    int detail;           /* 0 = full detail */
                          /* 1 = only consider voxels in beam path */
};

PLMDOSE_C_API
void
proton_dose_compute (
    Volume* dose_vol,
    Volume* ct_vol,
    Proton_dose_parms* parms
);

#endif
