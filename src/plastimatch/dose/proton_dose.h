/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _proton_dose_h_
#define _proton_dose_h_

#include "plmdose_config.h"
#include "threading.h"
#include "plm_path.h"

class Proton_Scene;
class Volume;

#define INDEX_OF(ijk, dim) \
    (((ijk[2] * dim[1] + ijk[1]) * dim[0]) + ijk[0])


class PLMDOSE_API Proton_dose_parms {
public:
    Proton_dose_parms ();
    ~Proton_dose_parms ();

    void parse_args (int argc, char** argv);

private:
    void parse_config (const char* config_fn);
    int set_key_val (const char* key, const char* val, int section);

public:
    /* [SETTINGS] */
    Threading threading;
    int debug;            /* 1 = debug mode */
    int detail;           /* 0 = full detail */
    char flavor;          /* Which algorithm? */
    float ray_step;       /* Uniform ray step size (mm) */
    float scale;          /* scale dose intensity */
                          /* 1 = only consider voxels in beam path */

    char input_fn[_MAX_PATH];  /* input:  patient volume */
    char output_fn[_MAX_PATH]; /* output: dose volume */

    Proton_Scene* scene;
};

PLMDOSE_C_API
Volume*
proton_dose_compute (Proton_dose_parms* parms);

#endif
