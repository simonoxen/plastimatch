/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _proton_dose_opts_h_
#define _proton_dose_opts_h_

class Proton_Scene;

class PLMDOSE_API Proton_Parms {
public:
    Proton_Parms ();
    ~Proton_Parms ();

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

#endif
