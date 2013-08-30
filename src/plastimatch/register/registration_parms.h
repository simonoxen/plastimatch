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

#define STAGE_TRANSFORM_NONE                0
#define STAGE_TRANSFORM_TRANSLATION         1
#define STAGE_TRANSFORM_VERSOR              2
#define STAGE_TRANSFORM_QUATERNION          3
#define STAGE_TRANSFORM_AFFINE              4
#define STAGE_TRANSFORM_BSPLINE             5
#define STAGE_TRANSFORM_VECTOR_FIELD        6
#define STAGE_TRANSFORM_ALIGN_CENTER        7

#define OPTIMIZATION_NO_REGISTRATION        0
#define OPTIMIZATION_AMOEBA                 1
#define OPTIMIZATION_RSG                    2
#define OPTIMIZATION_VERSOR                 3
#define OPTIMIZATION_LBFGS                  4
#define OPTIMIZATION_LBFGSB                 5
#define OPTIMIZATION_DEMONS                 6
#define OPTIMIZATION_STEEPEST               7
#define OPTIMIZATION_QUAT                   8
#define OPTIMIZATION_LIBLBFGS               9

#define IMPLEMENTATION_NONE                 0
#define IMPLEMENTATION_ITK                  1
#define IMPLEMENTATION_PLASTIMATCH          2

#define METRIC_NONE                         0
#define METRIC_MSE                          1
#define METRIC_MI                           2
#define METRIC_MI_MATTES                    3

#define IMG_OUT_FMT_AUTO                    0
#define IMG_OUT_FMT_DICOM                   1


class Plm_image;
class Registration_parms_private;
class Stage_parms;

class PLMREGISTER_API Registration_parms {
public:
    Registration_parms_private *d_ptr;
public:
    int num_stages;
    std::string moving_roi_fn;
    std::string fixed_roi_fn;
    int img_out_fmt;
    Plm_image_type img_out_type;
    char img_out_fn[_MAX_PATH];
    char xf_in_fn[_MAX_PATH];
    bool xf_out_itk;
    //char xf_out_fn[_MAX_PATH];
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
    std::list<Stage_parms*>& get_stages ();
    Stage_parms* append_stage ();
};

#endif
