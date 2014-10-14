/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _RTP_parms_h_
#define _RTP_parms_h_

#include "plmdose_config.h"
#include <string>
#include "plm_path.h"
#include "RTP_plan.h"
#include "threading.h"

class Plm_image;
class RTP_parms_private;
class RTP_plan;

class PLMDOSE_API RTP_parms
{
public:
    RTP_parms_private *d_ptr;
public:
    RTP_parms ();
    ~RTP_parms ();

    bool parse_args (int argc, char** argv);

public:
    RTP_plan::Pointer& get_plan ();

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
    char homo_approx;     /* Sigma approximation for homogeneous volume */
    float ray_step;       /* Uniform ray step size (mm) */
    float scale;          /* scale dose intensity */
                          /* 1 = only consider voxels in beam path */
    std::string input_ct_fn;    /* input:  patient volume */

    /* GCS FIX: Copy-paste with wed_parms.h */
};

#endif
