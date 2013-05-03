/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _proton_parms_h_
#define _proton_parms_h_

#include "plmdose_config.h"
#include <string>
#include "plm_path.h"
#include "proton_scene.h"
#include "threading.h"

class Plm_image;
class Proton_parms_private;
class Proton_scene;

class PLMDOSE_API Proton_parms
{
public:
    Proton_parms_private *d_ptr;
public:
    Proton_parms ();
    ~Proton_parms ();

    bool parse_args (int argc, char** argv);

public:
    Proton_scene::Pointer& get_scene ();
#if defined (commentout)
    void set_scene (Proton_scene *scene);
#endif

protected:
    void handle_end_of_section (int section);
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
    std::string input_ct_fn;  /* input:  patient volume */
    std::string output_dose_fn; /* output: dose volume */

    /* GCS FIX: Copy-paste with wed_parms.h */
};

#endif
