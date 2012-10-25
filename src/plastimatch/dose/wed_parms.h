/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _wed_parms_h_
#define _wed_parms_h_

#include "plmdose_config.h"
#include "sys/plm_path.h"

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
    bool have_ray_step;
    float ray_step;                 /* Uniform ray step size (mm) */
    char input_ct_fn[_MAX_PATH];    /* input:  patient volume */
    std::string input_dose_fn;      /* input:     dose volume */
    char output_ct_fn[_MAX_PATH];   /* output: patient volume */
    std::string output_dose_fn;     /* output:    dose volume */
    std::string rpl_vol_fn;         /* output: rpl volume */

    float src[3];
    float isocenter[3];
    float beam_res;

    float vup[3];
    int ires[2];
    bool have_ic;
    float ic[2];
    float ap_spacing[2];
    float ap_offset;
};

#endif
