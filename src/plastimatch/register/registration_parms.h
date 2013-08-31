/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _registration_parms_h_
#define _registration_parms_h_

#include "plmregister_config.h"
#include <list>
#include <string>
#include <ctype.h>
#include <stdlib.h>
#include "bspline.h"    /* for enums */
#include "plm_image_type.h"
#include "plm_path.h"
#include "pstring.h"
#include "threading.h"

class Plm_image;
class Registration_parms_private;
class Shared_parms;
class Stage_parms;

class PLMREGISTER_API Registration_parms {
public:
    Registration_parms_private *d_ptr;
public:
    int num_stages;
    int img_out_fmt;
    Plm_image_type img_out_type;
    char img_out_fn[_MAX_PATH];
    char xf_in_fn[_MAX_PATH];
    bool xf_out_itk;
    std::list<std::string> xf_out_fn;
    Pstring warped_landmarks_fn;
    Pstring fixed_landmarks_fn;
    Pstring moving_landmarks_fn;
    Pstring fixed_landmarks_list;
    Pstring moving_landmarks_list;
    char vf_out_fn[_MAX_PATH];
    char log_fn[_MAX_PATH];
    float default_value;           /* Replacement when out-of-view */
    int init_type;
    double init[12];

    /* for 4D and atlas */
    char moving_dir[_MAX_PATH];
    char fixed_dir[_MAX_PATH];
    char img_out_dir[_MAX_PATH];
    char vf_out_dir[_MAX_PATH];
    char moving_jobs[255][_MAX_PATH];
    char fixed_jobs[255][_MAX_PATH];
    int job_idx;
    int num_jobs;

public:
    Registration_parms();
    ~Registration_parms();
public:
    int set_command_string (const std::string& command_string);
    int set_key_val (const char* key, const char* val, int section);
    int parse_command_file (const char* options_fn);
    void set_job_paths (void);
public:
    const std::string& get_fixed_fn ();
    const std::string& get_moving_fn ();
    Shared_parms* get_shared_parms ();
    std::list<Stage_parms*>& get_stages ();
    Stage_parms* append_stage ();
};

#endif
