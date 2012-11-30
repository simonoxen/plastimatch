/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _proton_parms_h_
#define _proton_parms_h_

#include "plmdose_config.h"
#include <string>
#include "plm_path.h"
#include "threading.h"

class Plm_image;
class Proton_parms_private;
class Proton_Scene;

class PLMDOSE_API Proton_Parms {
public:
    Proton_Parms ();
    ~Proton_Parms ();

    void set_scene (Proton_Scene *scene);
    bool parse_args (int argc, char** argv);

private:
    void handle_end_of_section (int section);
    void parse_config (const char* config_fn);
    int set_key_val (const char* key, const char* val, int section);

public:
    Proton_parms_private *d_ptr;

public:
    /* [SETTINGS] */
    Threading threading;
    int debug;            /* 1 = debug mode */
    int detail;           /* 0 = full detail */
    char flavor;          /* Which algorithm? */
    float ray_step;       /* Uniform ray step size (mm) */
    float scale;          /* scale dose intensity */
                          /* 1 = only consider voxels in beam path */
    std::string input_ct_fn;  /* input:  patient volume */
    std::string output_dose_fn; /* output: dose volume */

    /* GCS FIX: Copy-paste with wed_parms.h */

    /* Scene (owned by caller) */
    Proton_Scene* scene;
};

#endif
