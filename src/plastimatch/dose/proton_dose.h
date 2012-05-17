/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _proton_dose_h_
#define _proton_dose_h_

#include "plmdose_config.h"
#include "threading.h"
#include "plm_path.h"

class Volume;

#define INDEX_OF(ijk, dim) \
    (((ijk[2] * dim[1] + ijk[1]) * dim[0]) + ijk[0])

class Proton_dose_parms {
public:
    Proton_dose_parms ();

    void parse_args (int argc, char** argv);
    void parse_config (const char* config_fn);
    int set_key_val (const char* key, const char* val, int section);

public:
    /* [SETTINGS] */
    Threading threading;
    char flavor;          /* Which algorithm? */
    float ray_step;       /* Uniform ray step size (mm) */
    float scale;          /* scale dose intensity */
    int detail;           /* 0 = full detail */
                          /* 1 = only consider voxels in beam path */
    char input_fn[_MAX_PATH];  /* input:  patient volume */
    char output_fn[_MAX_PATH]; /* output: dose volume */

    /* [BEAM] */
    char input_pep_fn[_MAX_PATH];   /* Proton energy profile */
    double src[3];         /* Beam source */
    double isocenter[3];

    /* [APERTURE] */
    double ap_offset;     /* distance from beam nozzle */
    double vup[3];        /* orientation */
    double ic [2];        /* center */
    int ires[2];          /* resolution (vox) */


    /* command line */
    int debug;
};

PLMDOSE_C_API
void
proton_dose_compute (
    Volume* dose_vol,
    Volume* ct_vol,
    Proton_dose_parms* parms
);

#endif
