/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _ion_parms_h_
#define _ion_parms_h_

#include "plmdose_config.h"
#include <string>
#include "plm_path.h"
#include "ion_plan.h"
#include "threading.h"

class Plm_image;
class Ion_parms_private;
class Ion_plan;

class PLMDOSE_API Ion_parms
{
public:
    Ion_parms_private *d_ptr;
public:
    Ion_parms ();
    ~Ion_parms ();

    bool parse_args (int argc, char** argv);

public:
    Ion_plan::Pointer& get_plan ();

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
    std::string input_ct_fn;    /* input:  patient volume */
    std::string output_dose_fn; /* output: dose volume */

    /* GCS FIX: Copy-paste with wed_parms.h */
};

#endif
