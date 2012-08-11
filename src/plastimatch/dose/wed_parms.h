/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _wed_parms_h_
#define _wed_parms_h_

#include "sys/plm_path.h"

class Plm_image;
class Proton_Scene;

class PLMDOSE_API Wed_Parms {
public:
    Wed_Parms ();
    ~Wed_Parms ();

    bool parse_args (int argc, char** argv);

private:
    void parse_config (const char* config_fn);
    int set_key_val (const char* key, const char* val, int section);

public:
    /* [SETTINGS] */
    int debug;
    float ray_step;       /* Uniform ray step size (mm) */
    char input_ct_fn[_MAX_PATH];    /* input:  patient volume */
    char input_dose_fn[_MAX_PATH];  /* input:     dose volume */
    char output_ct_fn[_MAX_PATH];   /* output: patient volume */
    char output_dose_fn[_MAX_PATH]; /* output:    dose volume */

    Plm_image* ct_vol;
    Plm_image* dose_vol;

    Proton_Scene* scene;
};

#endif
