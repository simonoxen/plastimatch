/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _Rt_parms_h_
#define _Rt_parms_h_

#include "plmdose_config.h"
#include <string>
#include "plm_path.h"
#include "rt_plan.h"
#include "threading.h"

class Plm_image;
class Rt_parms_private;
class Rt_plan;

class PLMDOSE_API Rt_parms
{
public:
    Rt_parms_private *d_ptr;
public:
    Rt_parms ();
    ~Rt_parms ();

    bool parse_args (int argc, char** argv);

public:
    Rt_plan::Pointer& get_plan ();

	/* Save parameters in the beam_storage */
	void save_beam_parameters(int i, int section);
	void print_verif(Rt_plan::Pointer plan);

protected:
    void handle_end_of_section (int section);
    void parse_config (const char* config_fn);
    int set_key_val (const char* key, const char* val, int section);

    /* GCS FIX: Copy-paste with wed_parms.h */
};

#endif
